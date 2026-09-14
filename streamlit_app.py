# ============================================================================
# ADVANCED RAG — STREAMLIT APP
# ============================================================================
#
# Same pipeline as the Hugging Face Space version, on a simpler host:
#   PDF -> chunking -> MiniLM embeddings -> FAISS + BM25 -> RRF
#        -> cross-encoder reranking -> evidence gate -> hosted LLM
#
# Retrieval runs right here, in Streamlit Community Cloud's normal
# container (CPU only, no GPU needed). The final answer-writing step
# calls Hugging Face's Inference Providers router instead of self-hosting
# a model, so there's no shared-GPU-scheduler dependency anywhere in this
# app -- that was the source of the repeated deploy failures on the
# previous host.
# ============================================================================

import os
import tempfile
import time
import traceback
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient

from rag_engine import AdvancedRAG


# ============================================================================
# CONFIGURATION
# ============================================================================

CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-120b",
]

MAX_CONTEXT_CHARS = 10000
MAX_TOKENS = 400

SYSTEM_PROMPT = (
    "You are a strict document question-answering assistant.\n"
    "Answer ONLY from the supplied document context.\n"
    "Do not use outside knowledge.\n"
    "Do not invent facts.\n"
    "If the context is insufficient, say: The documents do not provide "
    "enough information.\n"
    "Keep the answer concise and factual."
)


def get_hf_token():
    """Read HF_TOKEN from Streamlit Cloud secrets first (Settings -> Secrets
    on share.streamlit.io), falling back to a plain environment variable
    for local runs (`export HF_TOKEN=...` before `streamlit run`)."""
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


# ============================================================================
# CACHED, SHARED, EXPENSIVE RESOURCES
# ============================================================================
#
# @st.cache_resource loads these ONCE per running server process (not once
# per browser tab, not once per rerun) -- the equivalent of the module-scope
# BASE_RAG in the old Gradio app.
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding and reranker models...")
def load_shared_models():
    base = AdvancedRAG()
    return base.embed_tokenizer, base.embed_model, base.rerank_tokenizer, base.rerank_model


@st.cache_resource(show_spinner=False)
def get_inference_client():
    token = get_hf_token()
    if not token:
        return None
    return InferenceClient(token=token)


# ============================================================================
# GENERATION VIA HUGGING FACE INFERENCE PROVIDERS
# ============================================================================

def call_llm(context, question):
    """Try each candidate model on the router in turn. Returns
    (answer_or_None, error_message_or_None)."""

    client = get_inference_client()
    if client is None:
        return None, (
            "No Hugging Face token configured. Add HF_TOKEN under this "
            "app's Settings -> Secrets on Streamlit Community Cloud (or "
            "export HF_TOKEN before running locally)."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DOCUMENT CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}"},
    ]

    last_exc = None
    for model in CANDIDATE_MODELS:
        for attempt in range(2):
            try:
                completion = client.chat_completion(
                    messages=messages, model=model, max_tokens=MAX_TOKENS, temperature=0.0,
                )
                answer = (completion.choices[0].message.content or "").strip()
                if answer:
                    return answer, None
            except Exception as exc:
                last_exc = exc
                traceback.print_exc()
                time.sleep(2)

    return None, (
        "Every model on Hugging Face's Inference Providers router failed "
        f"for this request. Last error: {type(last_exc).__name__}: {last_exc}"
    )


# ============================================================================
# RETRIEVAL HELPERS
# ============================================================================

def format_sources(ranked):
    return "\n".join(
        f"- **{item['source']}**, page {item['page']} (reranker `{item['reranker_score']:.3f}`)"
        for item in ranked
    )


def build_context(ranked):
    parts = []
    remaining = MAX_CONTEXT_CHARS
    for index, item in enumerate(ranked, start=1):
        block = f"[Source {index} | {item['source']} | Page {item['page']}]\n{item['text']}"
        if len(block) > remaining:
            block = block[:remaining]
        parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def rebuild_index(uploaded_files, embed_tok, embed_model, rerank_tok, rerank_model):
    rag = AdvancedRAG(
        embed_tokenizer=embed_tok, embed_model=embed_model,
        rerank_tokenizer=rerank_tok, rerank_model=rerank_model,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="advanced_rag_"))
    pdf_paths = []
    for uploaded in uploaded_files:
        destination = temp_dir / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        pdf_paths.append(destination)

    summary = rag.build_index(pdf_paths)
    return rag, summary


def answer_question(rag, question):
    ranked, metrics = rag.retrieve(question)

    if not ranked:
        return {"status": "no_evidence", "question": question}

    sources_md = format_sources(ranked)

    if not metrics["supported"]:
        return {
            "status": "insufficient", "question": question,
            "sources_md": sources_md, "metrics": metrics,
        }

    context = build_context(ranked)
    answer, error = call_llm(context, question)

    return {
        "status": "error" if error else "answered",
        "question": question, "sources_md": sources_md, "metrics": metrics,
        "answer": answer, "error": error,
    }


# ============================================================================
# UI
# ============================================================================

st.set_page_config(page_title="Advanced RAG - Document Intelligence", page_icon="📚", layout="centered")

st.title("📚 Advanced RAG — Document Intelligence")
st.markdown(
    "**PDF → Transformer Embeddings → FAISS + BM25 → RRF → Reranker → "
    "Evidence Gate → hosted LLM**\n\n"
    "Retrieval (parsing, chunking, embedding, FAISS, BM25, reranking) runs "
    "on this app's own CPU. The final answer is written by a hosted model "
    "through Hugging Face's Inference Providers API, so answering doesn't "
    "depend on any GPU quota. Answers are only generated once the "
    "retrieval layer finds sufficient evidence."
)

embed_tok, embed_model, rerank_tok, rerank_model = load_shared_models()

if "rag" not in st.session_state:
    st.session_state.rag = None
if "index_summary" not in st.session_state:
    st.session_state.index_summary = None

st.header("1. Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True,
)

if st.button("Build / Rebuild Index", type="primary", disabled=not uploaded_files):
    with st.spinner("Parsing PDFs and building the FAISS + BM25 index..."):
        try:
            rag, summary = rebuild_index(uploaded_files, embed_tok, embed_model, rerank_tok, rerank_model)
            st.session_state.rag = rag
            st.session_state.index_summary = summary
        except Exception as exc:
            traceback.print_exc()
            st.error(f"Indexing failed: {type(exc).__name__}: {exc}")

if st.session_state.index_summary:
    summary = st.session_state.index_summary
    st.success(
        f"Index ready — {summary['documents']} document(s), "
        f"{summary['chunks']} chunks, {summary['faiss_vectors']} FAISS vectors "
        f"(embedding dim {summary['embedding_dimension']})."
    )
    for item in summary["pdf_stats"]:
        st.caption(f"`{item['file']}` — {item['pages']} pages, {item['chunks']} chunks")
    for item in summary.get("failures", []):
        st.warning(f"Skipped `{item['file']}`: {item['error']}")

st.header("2. Ask your documents")
questions_text = st.text_area(
    "One question per line",
    placeholder="What is self attention?\nWhat is positional encoding?",
    height=120,
)

if st.button("Ask Your Documents", type="primary"):
    if st.session_state.rag is None:
        st.error("Build the index before asking questions.")
    elif not questions_text.strip():
        st.error("Enter at least one question.")
    else:
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
        for number, question in enumerate(questions, start=1):
            with st.spinner(f"Answering question {number}/{len(questions)}..."):
                result = answer_question(st.session_state.rag, question)

            st.subheader(f"Question {number}: {result['question']}")

            if result["status"] == "no_evidence":
                st.info("No relevant evidence found in the uploaded documents.")
            elif result["status"] == "insufficient":
                m = result["metrics"]
                st.warning(
                    "**Insufficient evidence** — the uploaded documents don't "
                    "contain enough reliable evidence to answer this.\n\n"
                    f"Semantic score: `{m['semantic']:.3f}` | "
                    f"Evidence agreement: `{m['agreement']}`"
                )
                with st.expander("Retrieved sources"):
                    st.markdown(result["sources_md"])
            elif result["status"] == "error":
                st.error(f"Generation error: {result['error']}")
                with st.expander("Retrieved evidence"):
                    st.markdown(result["sources_md"])
            else:
                st.markdown(result["answer"])
                with st.expander(f"Sources (evidence agreement: {result['metrics']['agreement']})"):
                    st.markdown(result["sources_md"])

st.divider()
st.caption(
    "Embeddings: sentence-transformers/all-MiniLM-L6-v2 · Dense: FAISS · "
    "Sparse: BM25 · Fusion: RRF · Reranker: cross-encoder/ms-marco-MiniLM-L6-v2 "
    "· Generation: Hugging Face Inference Providers"
)
