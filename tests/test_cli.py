"""Tests for ragqa.cli.

The CLI wires modules together. Tests use a real chunker (with the
`make_pdf` fixture) but mock the embedder (no torch load) and LLM
(no HTTP). That keeps each test < 100ms but exercises the actual
file paths and command parsing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ragqa.cli import main


# ───────────────────────── fakes ───────────────────────────────────────────


def _patched_embedder(dim: int = 4):
    """Patch ragqa.embedding.SentenceTransformer so Embedder doesn't load
    PyTorch + the 80MB model in tests."""
    mock_st = MagicMock()
    mock_st.get_sentence_embedding_dimension.return_value = dim

    def encode_side_effect(texts, normalize_embeddings=True, **_):
        rng = np.random.RandomState(0)
        arr = rng.randn(len(texts), dim).astype(np.float32)
        if normalize_embeddings:
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr

    mock_st.encode.side_effect = encode_side_effect
    return mock_st


# ───────────────────────── ingest ──────────────────────────────────────────


def test_ingest_creates_index_from_single_pdf(make_pdf, tmp_path, capsys):
    pdf = make_pdf(["The cat sat on the mat. " * 20])
    out_dir = tmp_path / "myindex"

    with patch("ragqa.embedding.SentenceTransformer",
               return_value=_patched_embedder()):
        rc = main(["ingest", "-i", str(pdf), "-o", str(out_dir)])

    assert rc == 0
    assert (out_dir / "index.faiss").is_file()
    assert (out_dir / "chunks.json").is_file()
    assert (out_dir / "meta.json").is_file()

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_chunks"] >= 1


def test_ingest_creates_index_from_directory(make_pdf, tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    # Put two PDFs in a directory; the ingester should pick up both.
    p1 = make_pdf(["Doc one content. " * 20])
    p2 = make_pdf(["Doc two content. " * 20])
    (pdf_dir / "a.pdf").write_bytes(p1.read_bytes())
    (pdf_dir / "b.pdf").write_bytes(p2.read_bytes())

    out_dir = tmp_path / "idx"
    with patch("ragqa.embedding.SentenceTransformer",
               return_value=_patched_embedder()):
        rc = main(["ingest", "-i", str(pdf_dir), "-o", str(out_dir)])

    assert rc == 0
    chunks = json.loads((out_dir / "chunks.json").read_text())
    sources = {c["source_file"] for c in chunks}
    assert sources == {"a.pdf", "b.pdf"}


def test_ingest_with_no_pdfs_exits_error(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "idx"
    rc = main(["ingest", "-i", str(empty_dir), "-o", str(out_dir)])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "no pdf" in err or "not found" in err


# ───────────────────────── ask ─────────────────────────────────────────────


def _build_index_for_ask(make_pdf, tmp_path):
    """Helper: ingest a tiny PDF so the ask tests have something to load."""
    pdf = make_pdf(["The capital of France is Paris. " * 5])
    out = tmp_path / "idx"
    with patch("ragqa.embedding.SentenceTransformer",
               return_value=_patched_embedder()):
        rc = main(["ingest", "-i", str(pdf), "-o", str(out)])
    assert rc == 0
    return out


def test_ask_without_api_key_exits_error(make_pdf, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    idx = _build_index_for_ask(make_pdf, tmp_path)
    rc = main(["ask", "--index", str(idx), "What is the capital?"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "groq_api_key" in err


def test_ask_loads_index_and_prints_answer(make_pdf, tmp_path, capsys, monkeypatch):
    idx = _build_index_for_ask(make_pdf, tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "fake")

    # Mock both the embedder (for query encode) and the LLM HTTP layer.
    with patch("ragqa.embedding.SentenceTransformer",
               return_value=_patched_embedder()):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Paris [1]."}}]
        }
        with patch("ragqa.llm.requests.post", return_value=mock_resp):
            # Force min_score = 0 so the random-vector fake retrieval
            # actually returns chunks.
            rc = main([
                "ask", "--index", str(idx), "--min-score", "0",
                "What is the capital?",
            ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Paris" in out


def test_ask_prints_sources_section_when_citations_present(
    make_pdf, tmp_path, capsys, monkeypatch,
):
    idx = _build_index_for_ask(make_pdf, tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "fake")

    with patch("ragqa.embedding.SentenceTransformer",
               return_value=_patched_embedder()):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Paris [1]."}}]
        }
        with patch("ragqa.llm.requests.post", return_value=mock_resp):
            main([
                "ask", "--index", str(idx), "--min-score", "0",
                "What is the capital?",
            ])

    out = capsys.readouterr().out
    # Citation block should reference the source PDF and a page number.
    assert "Sources" in out or "sources" in out
    assert "page" in out.lower()


# ───────────────────────── help + parser ───────────────────────────────────


def test_help_exits_zero(capsys):
    """`ragqa --help` should exit 0 with usage info."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_no_subcommand_exits_nonzero(capsys):
    """Calling `ragqa` with no subcommand should error, not silently do nothing."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
