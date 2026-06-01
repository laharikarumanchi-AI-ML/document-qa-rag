# Document Q&A Assistant (RAG) — Spec

**Date:** 2026-06-01
**Owner:** Lahari Karumanchi
**Status:** Draft

## Goal

A retrieval-augmented chatbot that answers questions about a set of PDFs, with **every claim grounded in a citation** the user can click through to. The goal isn't "answer anything" — it's "answer from the documents, or admit you can't."

## Why this project

A general-purpose LLM asked "what does this PDF say about X?" will gladly fabricate a confident answer when X isn't in the document. RAG should make that failure mode go away — not by hoping the LLM is honest, but by structurally only giving it text that's actually in the docs, and forcing it to cite.

## Resume claim (target sentence)

> Built a retrieval-augmented Q&A assistant over PDFs: chunked with custom splitter, embedded with sentence-transformers, indexed in FAISS, answered with Llama-3.3-70B via Groq. Every answer includes the source page and offset so the user can verify; the system abstains when the answer isn't in the docs.

## Scope

### In scope

- PDF ingestion (one or many files) with page-aware chunking
- Local embedding via `sentence-transformers/all-MiniLM-L6-v2`
- FAISS index built once per corpus, saved to disk, loaded on query
- Top-k retrieval with cosine similarity
- Grounded answering via Groq Llama-3.3-70B, instructed to cite sources or abstain
- CLI: `ragqa ingest`, `ragqa ask`
- Streamlit demo: upload a PDF, ask questions, see citations inline
- Tests covering chunking, indexing, retrieval, and the citation-extraction post-processor

### Deliberately out of scope (for now)

- Multi-modal (images, tables in PDFs are extracted as text only)
- Re-ranking with a cross-encoder
- Conversation memory across questions (each question is independent)
- Multi-user / auth / persistence
- Fine-tuning embeddings
- Production deployment (local Streamlit only for v1)

## Architecture

```
   PDF files ─┐
              ▼
     ingest:  chunker → embedder → FAISS index (saved to indexes/<name>/)
              │
              ▼
     ask:    embed query → top-k retrieve → format prompt → Groq → parse citations → answer
```

### Modules

| File | Purpose | Public interface |
|---|---|---|
| `ragqa/chunking.py` | PDF → list of `Chunk(text, source_file, page, char_start, char_end)` | `chunk_pdf(path, *, target_chars, overlap_chars) -> list[Chunk]` |
| `ragqa/embedding.py` | sentence-transformers wrapper | `Embedder.encode(texts) -> np.ndarray` |
| `ragqa/index.py` | FAISS build / save / load | `Index.build(chunks, embedder)`; `Index.search(query_vec, k) -> list[(chunk, score)]` |
| `ragqa/retrieval.py` | Search + threshold filter | `retrieve(query, index, embedder, k, min_score) -> list[Chunk]` |
| `ragqa/answer.py` | Prompt construction + Groq call + citation post-process | `answer(query, chunks, llm) -> Answer(text, citations, abstained)` |
| `ragqa/llm.py` | Groq HTTP client (Retry-After-aware) | `GroqClient.chat(messages) -> str` |
| `ragqa/cli.py` | `ragqa ingest -i pdfs/ -o indexes/foo`<br>`ragqa ask --index indexes/foo "Q?"` | `main(argv)` |
| `demo/app.py` | Streamlit upload + chat UI | `streamlit run demo/app.py` |

## Key design decisions

### 1. No LangChain

Chunker, retriever, and answer loop are written directly (~200 lines total). Trade-off: more code; in exchange, every behavior — chunk boundaries, prompt format, citation extraction, retry semantics — is visible and testable.

### 2. Local embeddings, free LLM

sentence-transformers/all-MiniLM-L6-v2 (~80 MB, runs on CPU) for embeddings; Groq's free Llama-3.3-70B for the answer. Zero ongoing cost. The downside: lower-quality retrieval than OpenAI's `text-embedding-3-small`, but adequate for a portfolio-scale corpus (<10k chunks).

### 3. Page-aware chunking with character offsets

Each `Chunk` carries `page` and `char_start` / `char_end` within that page. The Streamlit demo can highlight the exact span used. Trade-off: more bookkeeping than a flat character chunker, but the citation precision is the whole point of the project.

### 4. Abstention on low-confidence retrieval

If the top retrieved chunk's cosine score is below `min_score` (default ~0.35), the answer module returns `Answer(text="I don't know based on the provided documents.", abstained=True)`. The model doesn't even get a chance to fabricate. This is the structural anti-hallucination claim.

### 5. Prompt format that forces citation

The system prompt instructs:

> Answer the user's question using ONLY the numbered context passages below. Cite each claim by appending `[N]` where N is the passage number. If the context doesn't contain the answer, respond with exactly: `I don't know based on the provided documents.`

After generation, `answer.py` parses `[N]` markers and maps them back to `(source_file, page, char_start, char_end)` for the caller.

## Testing

| Test file | What it covers |
|---|---|
| `tests/test_chunking.py` | PDF → chunks; page boundaries; overlap; offsets |
| `tests/test_index.py` | Build + save + load round-trip; search returns expected scores |
| `tests/test_retrieval.py` | Top-k ordering; threshold filtering |
| `tests/test_answer.py` | Prompt format; citation parsing; abstention case (mocked LLM) |
| `tests/test_llm.py` | Groq HTTP retry / backoff (mocked) |
| `tests/test_end_to_end.py` | One full ingest → ask cycle on a tiny test PDF with a scripted LLM |

## Open questions

- **Reasonable `min_score` default**? Tune empirically against a small handcrafted eval set.
- **Chunk size**? Start at 600 chars with 100 char overlap; revisit if retrieval feels too sparse or too noisy.
- **Demo PDFs**? Three small public-domain PDFs to vet for the demo (legal to redistribute, well-formed text layer). Candidates: a Project Gutenberg book chapter, an NIST publication, a public research paper.

## Done criteria

- All listed tests pass on `pytest -v`.
- `ragqa ingest` + `ragqa ask` produce correct citations on the vetted demo PDFs.
- Streamlit demo runs locally and shows citations inline.
- README documents install, ingest, ask, demo, and the design choices that differentiate this from a generic LangChain RAG tutorial.
- Portfolio's existing `/projects/document-qa-rag/` page is updated to point at the real repo + a real headline number (e.g., retrieval precision@5 on the demo set).
