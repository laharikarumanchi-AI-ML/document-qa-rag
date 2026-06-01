# Document Q&A Assistant (RAG)

[![tests](https://github.com/laharikarumanchi-AI-ML/document-qa-rag/actions/workflows/test.yml/badge.svg)](https://github.com/laharikarumanchi-AI-ML/document-qa-rag/actions/workflows/test.yml)

A retrieval-augmented chatbot that answers questions about a set of PDFs, with **every claim grounded in a citation** the user can click through to. The answer module abstains explicitly when the documents don't cover the question — the LLM never gets a chance to fabricate.

Built without LangChain. The chunker, retriever, prompt template, and citation parser are written directly so every behavior is visible and testable. **77 tests** across **8 modules**.

---

## How it works

```
   PDF ───▶ chunker ───▶ chunks ───▶ embedder ───▶ vectors
                                                      │
                                                      ▼
                                                  FAISS index ──save──▶ disk
                                                      │
                                                      ▼  (query time)
   question ──▶ embedder ──▶ retrieve ──▶ chunks ──▶ answer ──▶ Answer{text, citations, abstained}
                              │ (threshold:                       │
                              │  no chunk above                   │
                              │  → abstain)                       │
                              ▼                                   ▼
                            empty list                    "I don't know based on
                                │                          the provided documents."
                                ▼                          or grounded answer with [N] citations
                            abstain immediately
```

Two structural mechanisms keep hallucinations out:

- **Retrieval threshold.** If no chunk's cosine similarity exceeds `min_score` (default 0.35), the LLM is never called. The empty list is the abstention signal.
- **Prompt-enforced citation.** The system prompt instructs the model to either cite each claim with `[N]` markers (mapping back to the numbered context passages) or emit the exact abstention sentence. Post-processing detects the abstention sentence and propagates `abstained=True`.

---

## Quick start

```bash
# 1. Clone + install
git clone https://github.com/laharikarumanchi-AI-ML/document-qa-rag.git
cd document-qa-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Get a free Groq key at https://console.groq.com/keys
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_your_key_here

# 3. Ingest some PDFs (first run downloads the ~80MB embedding model)
ragqa ingest -i path/to/pdfs/ -o indexes/my-corpus

# 4. Ask questions
set -a && source .env && set +a
ragqa ask --index indexes/my-corpus "What does the paper say about X?"
```

Expected output:

```
The paper claims that X is correlated with Y at a level of 0.78 [1],
though the authors caution this may be confounded by Z [2].

Sources:
  [1] paper1.pdf, page 4
  [2] paper1.pdf, page 7
```

For the interactive web demo (Streamlit, local only):

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

---

## Module map

| File | Purpose | Public surface |
|---|---|---|
| [`ragqa/chunking.py`](ragqa/chunking.py) | PDF → page-aware chunks with character offsets | `chunk_pdf(path, target_chars, overlap_chars) -> list[Chunk]` |
| [`ragqa/embedding.py`](ragqa/embedding.py) | sentence-transformers wrapper. Lazy load. L2-normalized float32. | `Embedder().encode(texts)`, `.encode_query(text)`, `.dim` |
| [`ragqa/index.py`](ragqa/index.py) | FAISS inner-product index + parallel `list[Chunk]`. Save/load. | `Index.build(chunks, embedder)`, `.search(q, k)`, `.save(path)`, `.load(path)` |
| [`ragqa/retrieval.py`](ragqa/retrieval.py) | Top-k + threshold. The structural abstention mechanism. | `retrieve(query, *, index, embedder, k, min_score) -> list[(Chunk, float)]` |
| [`ragqa/llm.py`](ragqa/llm.py) | Groq HTTP client with Retry-After-aware backoff | `GroqClient.chat(messages) -> str` |
| [`ragqa/answer.py`](ragqa/answer.py) | Grounded prompt + LLM call + `[N]` citation parser | `answer(query, retrieved, llm) -> Answer{text, citations, abstained}` |
| [`ragqa/cli.py`](ragqa/cli.py) | `ragqa ingest` / `ragqa ask` | `ragqa --help` |
| [`demo/app.py`](demo/app.py) | Streamlit UI | `streamlit run demo/app.py` |

Each module is independently testable, mocked at module boundaries. See [`docs/spec.md`](docs/spec.md) for the original one-page design doc.

---

## Key design choices

### No LangChain

The whole pipeline is ~600 lines of straightforward Python across 8 files. Trade-off: more code than `LangChain.RetrievalQA`; in exchange, every behavior — chunk boundaries, prompt format, citation extraction, retry semantics, abstention plumbing — is visible at file paths you can grep for. Frameworks make the surface area opaque; writing it directly was the only way to know exactly what the system does when it doesn't return what I expected.

### Page-aware chunks with character offsets

Each chunk carries `(source_file, page, char_start, char_end)`. Chunks never span page boundaries. A citation = (file, page, span) — the UI highlights the exact region the LLM grounded its claim in. This is what gives "verifiable answer" actual teeth instead of being a vibes claim.

### Threshold-based abstention BEFORE the LLM

The classical RAG failure mode is: retriever returns weak matches → LLM dutifully answers from them anyway → answer is wrong but sounds confident. The fix is *structural*: filter chunks below `min_score` before the LLM call. If nothing crosses the bar, return the abstention message without calling the model. The LLM literally cannot fabricate from material it never sees. (`ragqa/retrieval.py` + the empty-list early return in `ragqa/answer.py`.)

### Prompt forces `[N]` citations or the exact abstention string

System prompt: "Cite each claim by appending [N]. If the context doesn't cover the question, respond with EXACTLY this sentence: I don't know based on the provided documents." After generation, `_parse_citations()` extracts the markers and maps back to chunks. If the model emits the abstention string, propagate `abstained=True`. The string is a constant referenced in both the prompt and the detector — single source of truth.

### Free local stack

`sentence-transformers/all-MiniLM-L6-v2` for embeddings (CPU, ~80MB). Groq's free Llama-3.3-70B for the LLM. Zero ongoing cost.

---

## Repo layout

```
ragqa/                    # The package
  chunking.py
  embedding.py
  index.py
  retrieval.py
  llm.py
  answer.py
  cli.py
  __main__.py             # `python -m ragqa` entry point

demo/
  app.py                  # Streamlit UI
  pdfs/                   # vetted demo PDFs go here (gitignored)

docs/
  spec.md                 # one-page design doc

tests/                    # pytest suite — 77 tests
  conftest.py             # make_pdf fixture (reportlab-generated PDFs)
  test_chunking.py        # 16
  test_embedding.py       # 7 + 1 slow
  test_index.py           # 9
  test_retrieval.py       # 10
  test_llm.py             # 7
  test_answer.py          # 17
  test_cli.py             # 8
  test_demo_imports.py    # 2
```

---

## Testing

```bash
pytest -v                  # 76 fast tests (~3 seconds)
pytest -m slow             # +1 real-model embedding test (downloads ~80MB)
```

Test pyramid:

- **Unit tests** mock at module boundaries: the embedder is mocked when testing the index; the LLM is mocked when testing answer; etc. Each unit's correctness is verified independently.
- **Integration tests** (the CLI tests in particular) use real chunker + real FAISS + mocked embedder + mocked LLM — so file paths, command parsing, and module wiring are exercised, while keeping each test <100ms.
- **The single slow test** uses the actual `all-MiniLM-L6-v2` model to verify the semantic-similarity contract still holds (`"cat sat on mat" ~ "feline rested on rug" > "cat sat on mat" ~ "stock prices surged"`). Cheap regression net for an embedding swap.

CI runs `pytest -v` on every PR via [`.github/workflows/test.yml`](.github/workflows/test.yml).

---

## Limitations

- **No re-ranking.** The retrieved chunks go straight into the prompt. A cross-encoder re-ranker would likely improve precision at high `min_score`; deliberately deferred.
- **Single-PDF UI.** The Streamlit demo handles one PDF per session. Multi-PDF would need either an upload list or persistent saved indexes; out of scope for v1.
- **No deploy yet.** The demo is local-only. Hugging Face Spaces is the obvious next step.
- **Threshold is one global number.** `min_score = 0.35` works for the all-MiniLM family but might be wrong for your corpus. Tune empirically with a small handcrafted eval set.
- **No conversation memory.** Each question is independent; no follow-up Q&A.
- **Eval numbers are still placeholder.** This README will be updated with real precision@5 on a vetted PDF set once the user runs the pipeline against one.

---

## What's next

In rough priority order:

1. **Run the CLI on a real PDF corpus** and record headline numbers in this README (precision@5 on a handcrafted eval set).
2. **Deploy the Streamlit demo to Hugging Face Spaces** (vetted PDFs only — no arbitrary uploads in v1).
3. **Update the portfolio's `/projects/document-qa-rag/` page** to link here with the real headline.
4. **Add a cross-encoder re-ranker** as an optional second-stage pass.
5. **Conversation memory** — let follow-up questions reference earlier ones.

---

## Acknowledgments

- **sentence-transformers** for the local embedding stack.
- **Groq** for the free Llama-3.3-70B tier that makes the LLM side cost-free.
- The Retry-After-aware backoff pattern in `ragqa/llm.py` is ported from the [`superpowers` data-analysis agent project](https://github.com/laharikarumanchi-AI-ML/superpowers/blob/main/agent/llm_client.py), where it was battle-tested against the InfiAgent-DABench runs.
- This project was paired with Claude Code through a TDD-first workflow: 8 sequential PRs, each module gated on its own tests + CI. The implementation decisions, design choices, and "no LangChain" reflections are mine; the typing fingers were partly silicon.

<!--
## What I'd do differently
TODO (Lahari): a short, honest reflection paragraph once you've run this against a real
PDF corpus. Raw material to consider:

- The `min_score = 0.35` default is a guess until you measure precision/recall on a
  handcrafted eval set. The "tune empirically" caveat above isn't a future task — it's
  the next concrete thing to do.
- The single-PDF Streamlit demo is a v1 compromise. If recruiters' questions ever land
  on "could you handle a multi-doc workflow?" you can point at the planned change in
  the limitations section.
- "Built without LangChain" only signals depth if you can defend why each replacement
  decision matters. The README does this for citations + abstention, but the chunker's
  boundary-snapping is also a real choice (paragraph > sentence > word). Worth a
  mention if you discuss this with anyone technical.
-->
