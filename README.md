# Document Q&A Assistant (RAG)

[![tests](https://github.com/laharikarumanchi-AI-ML/document-qa-rag/actions/workflows/test.yml/badge.svg)](https://github.com/laharikarumanchi-AI-ML/document-qa-rag/actions/workflows/test.yml)

A retrieval-augmented chatbot that answers questions about PDFs, with **every claim grounded in a citation** the user can verify. Built without LangChain — chunker, retriever, and answer loop are written directly so every behavior is visible and testable.

> 🚧 **Status: scaffolded.** Modules and tests are being implemented PR-by-PR. See [`docs/spec.md`](docs/spec.md) for the design and [open PRs](https://github.com/laharikarumanchi-AI-ML/document-qa-rag/pulls) for in-flight work.

---

## Stack

| Piece | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Runs on CPU, no API cost, ~80 MB |
| Vector store | FAISS (local, on disk) | Fast cosine search, no service to run |
| LLM | Groq Llama-3.3-70B | Free tier, fast (~500 tok/s) |
| PDF parser | `pdfplumber` | Preserves page structure + character positions |
| UI | Streamlit | Local demo with inline citation highlighting |
| Framework | None | The agent loop is ~200 lines of straight Python — see spec |

---

## Quick start (once implementation lands)

```bash
# Install
git clone https://github.com/laharikarumanchi-AI-ML/document-qa-rag.git
cd document-qa-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Get a free Groq key at https://console.groq.com/keys
cp .env.example .env
# Edit .env, add GROQ_API_KEY=gsk_...

# Ingest some PDFs (creates ./indexes/my-corpus/)
ragqa ingest -i path/to/pdfs/ -o indexes/my-corpus

# Ask questions
set -a && source .env && set +a
ragqa ask --index indexes/my-corpus "What does the paper say about X?"

# Interactive demo (Streamlit)
pip install -e ".[demo]"
streamlit run demo/app.py
```

---

## What's next

See [`docs/spec.md`](docs/spec.md) for the planned module breakdown and design choices. Implementation lands one PR per module on this repo.
