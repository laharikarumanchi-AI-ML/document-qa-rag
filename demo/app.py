"""Streamlit demo for ragqa.

Upload a PDF, ask a question, see an answer with citations to the
exact page. Same modules as the CLI; same abstention semantics.

Run locally:
    pip install -e ".[demo]"
    export GROQ_API_KEY=gsk_...
    streamlit run demo/app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from ragqa.answer import answer
from ragqa.chunking import chunk_pdf
from ragqa.embedding import Embedder
from ragqa.index import Index
from ragqa.llm import GroqClient
from ragqa.retrieval import DEFAULT_K, DEFAULT_MIN_SCORE, retrieve


# ───────────────────────── helpers ─────────────────────────────────────────


@st.cache_resource
def get_embedder() -> Embedder:
    """The embedder is heavy (PyTorch + ~80MB of weights). Cache it for
    the lifetime of the Streamlit server process."""
    return Embedder()


def _build_index(uploaded_file) -> Index | None:
    """Write upload to a temp file, chunk + embed + index, return Index.
    Returns None if no text could be extracted."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded_file.read())
        tmp_path = Path(f.name)
    try:
        chunks = chunk_pdf(tmp_path)
        if not chunks:
            return None
        embedder = get_embedder()
        return Index.build(chunks, embedder)
    finally:
        tmp_path.unlink(missing_ok=True)


# ───────────────────────── page ────────────────────────────────────────────


st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)
st.title("📄 Document Q&A")
st.caption(
    "Upload a PDF. Ask a question. The agent answers ONLY from the document "
    "and cites the page. If your question isn't in the document, it says so."
)


with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("PDF", type="pdf")

    if uploaded is not None:
        st.caption(f"**{uploaded.name}** — {uploaded.size // 1024} KB")
        # Re-index whenever a new file lands or its name changes
        if st.session_state.get("loaded_pdf") != uploaded.name:
            with st.spinner("Chunking + embedding..."):
                index = _build_index(uploaded)
            if index is None or index.size == 0:
                st.error("No extractable text in this PDF.")
                st.stop()
            st.session_state["index"] = index
            st.session_state["loaded_pdf"] = uploaded.name
            st.success(f"Indexed {index.size} chunks.")

    st.divider()
    st.header("Retrieval settings")
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10,
                      value=DEFAULT_K)
    min_score = st.slider(
        "Minimum cosine score",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_MIN_SCORE, step=0.05,
        help="If no chunk exceeds this score, the agent abstains. "
             "Lower = more recall (and more hallucination risk); higher "
             "= more abstentions but more grounded answers.",
    )
    model = st.selectbox("LLM",
                         ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])


# ───────────────────────── ask ─────────────────────────────────────────────


question = st.text_input(
    "Your question",
    placeholder="What does this document say about ...?",
)
ask_clicked = st.button("Ask", type="primary",
                        disabled=question == "" or "index" not in st.session_state)

if "index" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
elif ask_clicked:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set in this environment. "
            "Free key at https://console.groq.com/keys."
        )
    else:
        embedder = get_embedder()
        with st.spinner("Retrieving + asking..."):
            try:
                retrieved = retrieve(
                    question,
                    index=st.session_state["index"],
                    embedder=embedder,
                    k=top_k,
                    min_score=min_score,
                )
                llm = GroqClient(api_key=api_key, model=model)
                result = answer(question, retrieved=retrieved, llm=llm)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        # ─── render ───
        if result.abstained:
            st.info(f"💭 {result.text}")
        else:
            st.markdown("### Answer")
            st.write(result.text)

            if result.citations:
                st.markdown("### Sources")
                for i, chunk in enumerate(result.citations, start=1):
                    with st.expander(
                        f"[{i}] {chunk.source_file}, page {chunk.page}"
                    ):
                        st.write(chunk.text)
                        st.caption(
                            f"Chars {chunk.char_start}–{chunk.char_end} on "
                            f"page {chunk.page}"
                        )

        # Debug expander — useful for tuning retrieval
        with st.expander("Show all retrieved chunks (debug)"):
            if not retrieved:
                st.caption("(No chunks crossed the threshold.)")
            else:
                for i, (chunk, score) in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**[{i}]** {chunk.source_file}, page {chunk.page} "
                        f"— cosine {score:.3f}"
                    )
                    preview = chunk.text[:280]
                    if len(chunk.text) > 280:
                        preview += "..."
                    st.text(preview)
