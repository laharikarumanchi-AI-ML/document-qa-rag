"""Tests for ragqa.index.

The Index wraps FAISS + a parallel list of Chunks. All tests use a mocked
Embedder so they're fast and deterministic — we test the FAISS plumbing
and serialization, not the embedder's quality (that's test_embedding.py's
job).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragqa.chunking import Chunk
from ragqa.index import Index


# ───────────────────────── fakes / helpers ─────────────────────────────────


def _fake_embedder(dim: int = 4, name: str = "fake-model"):
    """A MagicMock that quacks like an Embedder. Returns vectors derived
    deterministically from the input text so we can reason about search
    results."""
    emb = MagicMock()
    emb.dim = dim
    emb.model_name = name

    def encode(texts):
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        # Map each text to a basis-aligned vector for predictable similarity.
        # Texts containing "cat" → strongly e0; "dog" → e1; "fish" → e2.
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t_low = t.lower()
            if "cat" in t_low:
                out[i, 0] = 1.0
            elif "dog" in t_low:
                out[i, 1] = 1.0
            elif "fish" in t_low:
                out[i, 2] = 1.0
            else:
                out[i, 3] = 1.0
        return out

    def encode_query(text):
        return encode([text])[0]

    emb.encode = MagicMock(side_effect=encode)
    emb.encode_query = MagicMock(side_effect=encode_query)
    return emb


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(text=t, source_file="x.pdf", page=1 + i, char_start=0, char_end=len(t))
        for i, t in enumerate(texts)
    ]


# ───────────────────────── build / size ────────────────────────────────────


def test_build_returns_index_with_correct_size():
    emb = _fake_embedder()
    idx = Index.build(_chunks(["cat", "dog", "fish"]), emb)
    assert idx.size == 3


def test_build_with_no_chunks_creates_empty_index():
    emb = _fake_embedder()
    idx = Index.build([], emb)
    assert idx.size == 0
    assert idx.search(np.zeros(4, dtype=np.float32), k=5) == []


def test_build_calls_embedder_once_with_all_texts():
    """We batch-encode at build time, not per-chunk. Matters for perf."""
    emb = _fake_embedder()
    chunks = _chunks(["cat", "dog", "fish"])
    Index.build(chunks, emb)
    assert emb.encode.call_count == 1
    # The single call should have been given all 3 texts
    call_args = emb.encode.call_args
    assert call_args[0][0] == ["cat", "dog", "fish"]


# ───────────────────────── search ──────────────────────────────────────────


def test_search_returns_top_k_in_score_order():
    emb = _fake_embedder()
    chunks = _chunks(["cat thing", "dog thing", "fish thing"])
    idx = Index.build(chunks, emb)

    # Query "cat" → vector e0 → exact match on chunk 0
    q = emb.encode_query("cat")
    results = idx.search(q, k=2)
    assert len(results) == 2
    # Top result is the cat chunk
    assert results[0][0].text == "cat thing"
    # Score is 1.0 (perfect match) within float tolerance
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    # Scores are non-increasing
    assert results[0][1] >= results[1][1]


def test_search_with_k_larger_than_index_size():
    emb = _fake_embedder()
    chunks = _chunks(["cat", "dog"])
    idx = Index.build(chunks, emb)
    q = emb.encode_query("cat")
    results = idx.search(q, k=10)
    assert len(results) == 2  # capped at index size, no crash


def test_search_returns_chunks_not_indices():
    """The result tuples carry the actual Chunk object — the citation
    pipeline can map back to (file, page, offsets) without a separate
    lookup."""
    emb = _fake_embedder()
    chunks = _chunks(["cat thing"])
    idx = Index.build(chunks, emb)
    q = emb.encode_query("cat")
    [(chunk, score)] = idx.search(q, k=1)
    assert isinstance(chunk, Chunk)
    assert chunk.source_file == "x.pdf"
    assert chunk.page == 1


# ───────────────────────── save / load round-trip ──────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path):
    emb = _fake_embedder()
    chunks = _chunks(["cat thing", "dog thing", "fish thing"])
    idx = Index.build(chunks, emb)

    idx.save(tmp_path / "myindex")

    # Verify the on-disk layout matches the spec
    assert (tmp_path / "myindex" / "index.faiss").is_file()
    assert (tmp_path / "myindex" / "chunks.json").is_file()
    assert (tmp_path / "myindex" / "meta.json").is_file()

    loaded = Index.load(tmp_path / "myindex")
    assert loaded.size == 3

    # Same chunks come out (Chunk is frozen, so dataclass equality works)
    q = emb.encode_query("dog")
    original_results = idx.search(q, k=3)
    loaded_results = loaded.search(q, k=3)
    assert [c.text for c, _ in original_results] == [c.text for c, _ in loaded_results]


def test_meta_records_embedder_model_and_dim(tmp_path: Path):
    """Saved metadata lets a loader sanity-check that the same embedder
    is being used at query time as was used at build time."""
    import json

    emb = _fake_embedder(dim=4, name="fake-model")
    idx = Index.build(_chunks(["cat"]), emb)
    idx.save(tmp_path / "myindex")

    meta = json.loads((tmp_path / "myindex" / "meta.json").read_text())
    assert meta["embedder_model"] == "fake-model"
    assert meta["dim"] == 4
    assert meta["n_chunks"] == 1


def test_search_score_for_orthogonal_query_is_near_zero():
    """Sanity: a query that's orthogonal to every chunk's vector should
    get scores ~ 0, not negative noise. (Our fake vectors are 1.0 or 0.0.)"""
    emb = _fake_embedder()
    chunks = _chunks(["cat", "dog"])  # vectors e0 and e1
    idx = Index.build(chunks, emb)
    # Query e2 (orthogonal to both)
    q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    results = idx.search(q, k=2)
    for _, score in results:
        assert score == pytest.approx(0.0, abs=1e-5)
