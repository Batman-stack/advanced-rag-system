# Advanced RAG — Document Intelligence

A hybrid PDF Retrieval-Augmented Generation app: upload PDFs, ask questions,
get answers grounded strictly in the retrieved passages (or a clear
"insufficient evidence" instead of a guess).

## Architecture

```
PDF -> PyPDF -> chunking -> MiniLM embeddings -> FAISS (dense) + BM25 (sparse)
     -> Reciprocal Rank Fusion -> cross-encoder reranking -> evidence gate
     -> hosted LLM (Hugging Face Inference Providers)
```

Retrieval (parsing, chunking, embeddings, FAISS, BM25, reranking) runs
locally in this app's own process — plain CPU, no GPU needed. The final
answer-writing step calls a hosted model through Hugging Face's Inference
Providers API instead of loading an LLM locally, so the app has no GPU
dependency at all. A short list of candidate models is tried in order so
one being temporarily unavailable doesn't break the app.

## Run locally

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # a Hugging Face access token
streamlit run streamlit_app.py
```

Get a token at https://huggingface.co/settings/tokens — a classic
**Write** token is simplest; if you use a fine-grained token instead,
enable "Make calls to Inference Providers".

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a **public** GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click **New app**, pick the repo/branch, set the main file path to
   `streamlit_app.py`, click **Deploy**.
4. Once it's deployed, open **Settings -> Secrets** for the app and add:
   ```toml
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   ```
5. The app restarts automatically after saving a secret. Open the app's
   `https://<something>.streamlit.app` URL — that's the link for your
   resume/portfolio/GitHub README.

No GPU, no dynamic hardware scheduling, no export-control-style hardware
tiers to configure — Community Cloud always runs on plain CPU, and this
app never needed a GPU in the first place.

## Notes

- Community Cloud's free tier gives ~1GB RAM per app and puts apps to
  sleep after long stretches with no visitors (the next visitor sees a
  "waking up" screen for a few seconds — normal, not a bug).
- Uploaded PDFs and the built index live only in that browser session's
  memory; they aren't persisted between restarts.
