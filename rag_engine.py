
from __future__ import annotations

import re
import traceback
from pathlib import Path

import faiss
import numpy as np
import torch

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================================
# MODEL IDS
# ============================================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# ============================================================================
# RETRIEVAL CONFIG
# ============================================================================

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

DENSE_K = 20
BM25_K = 20

RERANK_K = 12
SOURCE_K = 5

RRF_K = 60

# Cosine-similarity floor for the bi-encoder signal used in the grounding gate.
GROUNDING_THRESHOLD = 0.20
# Cross-encoder (ms-marco) logits are unbounded but, empirically, a positive
# score is a reasonably reliable "this passage is actually relevant" signal.
# It's a supplementary path into "supported", not a replacement for the
# semantic-score checks below.
RERANKER_GROUNDING_THRESHOLD = 0.0


# ============================================================================
# EMBEDDING UTILS
# ============================================================================

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def normalize_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


# ============================================================================
# RAG CLASS
# ============================================================================

class AdvancedRAG:

    def __init__(
        self,
        embed_tokenizer=None,
        embed_model=None,
        rerank_tokenizer=None,
        rerank_model=None,
    ):
        self.device = torch.device("cpu")

        self.embed_tokenizer = embed_tokenizer or AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.embed_model = embed_model or AutoModel.from_pretrained(EMBED_MODEL)
        self.embed_model.to(self.device)
        self.embed_model.eval()

        self.rerank_tokenizer = rerank_tokenizer or AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.rerank_model = rerank_model or AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.rerank_model.to(self.device)
        self.rerank_model.eval()

        self.index = None
        self.bm25 = None
        self.chunks = []
        self.embeddings = None

    # =====================================================================
    # TOKENIZATION
    # =====================================================================

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    # =====================================================================
    # PDF EXTRACTION
    # =====================================================================

    def extract_pdf(self, path):
        """Extract per-page text. Raises on a totally unreadable PDF so the
        caller (build_index) can skip just that one file instead of losing
        the whole batch."""
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append({"page": page_number, "text": text})
        return pages

    # =====================================================================
    # CHUNKING
    # =====================================================================

    def chunk_pages(self, path, pages):
        path = Path(path)
        chunks = []

        for page in pages:
            text = page["text"]
            start = 0

            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))

                if end < len(text):
                    cuts = [
                        text.rfind(". ", start, end),
                        text.rfind("? ", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("; ", start, end),
                    ]
                    best = max(cuts)
                    if best > start + int(CHUNK_SIZE * 0.55):
                        end = best + 1

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append({
                        "text": chunk,
                        "source": path.name,
                        "page": int(page["page"]),
                        "chunk_id": len(chunks),
                    })

                if end >= len(text):
                    break
                start = max(end - CHUNK_OVERLAP, start + 1)

        return chunks

    # =====================================================================
    # EMBEDDING
    # =====================================================================

    @torch.inference_mode()
    def encode(self, texts, batch_size=16):
        vectors = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            inputs = self.embed_tokenizer(
                batch, padding=True, truncation=True, max_length=384,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.embed_model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = normalize_embeddings(pooled)

            vectors.append(pooled.cpu().numpy())

        return np.concatenate(vectors, axis=0).astype("float32")

    # =====================================================================
    # INDEX BUILD
    # =====================================================================

    def build_index(self, pdf_paths):
        all_chunks = []
        pdf_stats = []
        failures = []
        documents = 0

        for pdf_path in pdf_paths:
            # FIX: one corrupted/unreadable PDF no longer aborts the whole
            # batch — it's recorded and skipped, the rest still get indexed.
            try:
                pages = self.extract_pdf(pdf_path)
            except Exception as exc:
                failures.append({"file": Path(pdf_path).name, "error": f"{type(exc).__name__}: {exc}"})
                continue

            if pages:
                documents += 1

            chunks = self.chunk_pages(pdf_path, pages)
            all_chunks.extend(chunks)

            pdf_stats.append({
                "file": Path(pdf_path).name,
                "pages": len(pages),
                "chunks": len(chunks),
            })

        if not all_chunks:
            detail = "; ".join(f"{f['file']}: {f['error']}" for f in failures) if failures else ""
            raise ValueError(
                "No extractable text found in any uploaded PDF. Scanned/"
                "image-only PDFs need OCR first."
                + (f" Per-file errors: {detail}" if detail else "")
            )

        texts = [item["text"] for item in all_chunks]
        embeddings = self.encode(texts)
        dimension = int(embeddings.shape[1])

        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        tokens = [self.tokenize(text) for text in texts]
        sparse_index = BM25Okapi(tokens)

        self.index = faiss_index
        self.embeddings = embeddings
        self.chunks = all_chunks
        self.bm25 = sparse_index

        return {
            "documents": documents,
            "chunks": len(all_chunks),
            "embedding_dimension": dimension,
            "faiss_vectors": int(faiss_index.ntotal),
            "pdf_stats": pdf_stats,
            "failures": failures,
        }

    # =====================================================================
    # DENSE RETRIEVAL
    # =====================================================================

    def dense(self, query, k=DENSE_K):
        if self.index is None or self.index.ntotal == 0:
            return []

        q = self.encode([query], batch_size=1)
        k = min(max(1, int(k)), self.index.ntotal)
        scores, ids = self.index.search(q, k)

        results = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            idx = int(idx)
            if idx < 0:
                continue
            results.append({"index": idx, "dense_score": float(score), "dense_rank": rank})

        return results

    # =====================================================================
    # SPARSE RETRIEVAL
    # =====================================================================

    def sparse(self, query, k=BM25_K):
        if self.bm25 is None or not self.chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype="float32")
        if len(scores) == 0:
            return []

        k = min(max(1, int(k)), len(scores))
        order = np.argsort(scores)[::-1][:k]

        return [
            {"index": int(idx), "bm25_score": float(scores[idx]), "bm25_rank": rank}
            for rank, idx in enumerate(order, start=1)
        ]

    # =====================================================================
    # RRF
    # =====================================================================

    def rrf(self, dense_results, sparse_results):
        merged = {}

        for item in dense_results:
            idx = item["index"]
            row = merged.setdefault(idx, {
                "index": idx, "rrf_score": 0.0, "dense_score": 0.0, "bm25_score": 0.0,
            })
            row["rrf_score"] += 1.0 / (RRF_K + item["dense_rank"])
            row["dense_score"] = item["dense_score"]

        for item in sparse_results:
            idx = item["index"]
            row = merged.setdefault(idx, {
                "index": idx, "rrf_score": 0.0, "dense_score": 0.0, "bm25_score": 0.0,
            })
            row["rrf_score"] += 1.0 / (RRF_K + item["bm25_rank"])
            row["bm25_score"] = item["bm25_score"]

        return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

    # =====================================================================
    # CROSS-ENCODER RERANKING
    # =====================================================================

    @torch.inference_mode()
    def rerank(self, query, candidates):
        if not candidates:
            return []

        queries = [query for _ in candidates]
        documents = [self.chunks[item["index"]]["text"] for item in candidates]

        encoded = self.rerank_tokenizer(
            queries, documents, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.rerank_model(**encoded)
        logits = outputs.logits

        if logits.ndim == 2:
            scores = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
        else:
            scores = logits.reshape(-1)

        scores = scores.detach().cpu().numpy()

        results = []
        for item, score in zip(candidates, scores):
            chunk = self.chunks[item["index"]]
            results.append({
                **item,
                "reranker_score": float(score),
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            })

        results.sort(key=lambda item: item["reranker_score"], reverse=True)
        return results[:SOURCE_K]

    # =====================================================================
    # COMPLETE RETRIEVAL PIPELINE
    # =====================================================================

    def retrieve(self, query):
        dense_results = self.dense(query)
        sparse_results = self.sparse(query)
        fused = self.rrf(dense_results, sparse_results)

        empty_metrics = {"supported": False, "semantic": 0.0, "reranker": 0.0, "agreement": 0}

        if not fused:
            return [], dict(empty_metrics)

        candidates = fused[:RERANK_K]
        ranked = self.rerank(query, candidates)

        if not ranked:
            return [], dict(empty_metrics)

        # FIX: compute the semantic (bi-encoder) signal from the *raw* dense
        # retrieval results, not from the post-rerank top-K. Any chunk that
        # RRF merged in from BM25 alone keeps its "dense_score" at the
        # merge default of 0.0 (it was never in the dense top-k). If the
        # cross-encoder's final top-K happens to end up dominated by such
        # chunks, taking max(dense_score) over *that* list can read 0.0
        # even though the bi-encoder found a strong match elsewhere in the
        # corpus — understating how well-grounded the answer really was.
        semantic_score = max((item["dense_score"] for item in dense_results), default=0.0)
        reranker_top_score = max(item["reranker_score"] for item in ranked)

        agreement = sum(
            1 for item in ranked[:5]
            if item["dense_score"] >= GROUNDING_THRESHOLD
        )

        supported = bool(
            semantic_score >= 0.30
            or (semantic_score >= GROUNDING_THRESHOLD and agreement >= 2)
            or reranker_top_score >= RERANKER_GROUNDING_THRESHOLD
        )

        return ranked, {
            "supported": supported,
            "semantic": float(semantic_score),
            "reranker": float(reranker_top_score),
            "agreement": int(agreement),
        }


# ============================================================================
# END
# ============================================================================
