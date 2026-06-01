"""Tests for ragqa.retrieval.

`retrieve()` is the function that turns "a question + an index" into
"the small set of chunks the LLM is allowed to ground its answer in."
The abstention threshold here is the *structural* anti-hallucination
mechanism — if no chunk crosses it, the LLM doesn't even get a chance
to fabricate.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ragqa.chunking import Chunk
from ragqa.retrieval import retrieve, DEFAULT_K, DEFAULT_MIN_SCORE


def _chunk(text: str, page: int = 1) -> Chunk:
    return Chunk(text=text, source_file="x.pdf", page=page,
                 char_start=0, char_end=len(text))


def _fake_index(results: list[tuple[Chunk, float]], model_name: str = "fake-model"):
    """Mock Index that returns `results` from search and reports `model_name`."""
    idx = MagicMock()
    idx.embedder_model = model_name
    idx.search.return_value = list(results)
    return idx


def _fake_embedder(model_name: str = "fake-model"):
    emb = MagicMock()
    emb.model_name = model_name
    emb.encode_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return emb


# ───────────────────────── threshold behavior ─────────────────────────────


def test_retrieve_filters_out_chunks_below_threshold():
    """Chunks with score < min_score must be excluded — they're noise."""
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    idx = _fake_index([(a, 0.80), (b, 0.40), (c, 0.10)])
    emb = _fake_embedder()
    out = retrieve("q?", index=idx, embedder=emb, min_score=0.35)
    texts = [chunk.text for chunk, _ in out]
    assert texts == ["a", "b"]  # c (0.10) drops out


def test_retrieve_returns_empty_when_all_below_threshold():
    """No chunk crosses the threshold → abstention case. Empty result is the
    signal to the answer module to say 'I don't know'."""
    a, b = _chunk("a"), _chunk("b")
    idx = _fake_index([(a, 0.20), (b, 0.10)])
    emb = _fake_embedder()
    out = retrieve("q?", index=idx, embedder=emb, min_score=0.35)
    assert out == []


def test_retrieve_returns_empty_when_index_is_empty():
    idx = _fake_index([])
    emb = _fake_embedder()
    out = retrieve("q?", index=idx, embedder=emb)
    assert out == []


def test_retrieve_returns_at_most_k():
    chunks = [_chunk(t) for t in "abcde"]
    idx = _fake_index([(c, 0.9 - 0.05 * i) for i, c in enumerate(chunks)])
    emb = _fake_embedder()
    out = retrieve("q?", index=idx, embedder=emb, k=3, min_score=0.0)
    assert len(out) == 3


def test_retrieve_preserves_score_ordering():
    """search() already orders results; retrieve must not re-order."""
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    idx = _fake_index([(a, 0.9), (b, 0.7), (c, 0.5)])
    emb = _fake_embedder()
    out = retrieve("q?", index=idx, embedder=emb, k=3, min_score=0.0)
    assert [s for _, s in out] == [0.9, 0.7, 0.5]


# ───────────────────────── embedder-model mismatch ────────────────────────


def test_retrieve_raises_on_embedder_model_mismatch():
    """Silent model swaps tank retrieval quality without error.
    Raise loudly so the failure is observable."""
    idx = _fake_index([], model_name="model-A")
    emb = _fake_embedder(model_name="model-B")
    with pytest.raises(ValueError, match="model-A.*model-B"):
        retrieve("q?", index=idx, embedder=emb)


def test_retrieve_passes_when_models_match():
    """Same model name on both sides → no error, normal retrieval."""
    a = _chunk("a")
    idx = _fake_index([(a, 0.9)], model_name="some-model")
    emb = _fake_embedder(model_name="some-model")
    out = retrieve("q?", index=idx, embedder=emb)
    assert len(out) == 1


# ───────────────────────── plumbing ────────────────────────────────────────


def test_retrieve_calls_embedder_encode_query_with_text():
    idx = _fake_index([])
    emb = _fake_embedder()
    retrieve("what is X?", index=idx, embedder=emb)
    emb.encode_query.assert_called_once_with("what is X?")


def test_retrieve_passes_k_through_to_index_search():
    idx = _fake_index([])
    emb = _fake_embedder()
    retrieve("q?", index=idx, embedder=emb, k=11)
    # search should have been called with the same k we asked for
    _, kwargs = idx.search.call_args
    assert kwargs.get("k") == 11 or idx.search.call_args[0][1] == 11


# ───────────────────────── defaults ────────────────────────────────────────


def test_defaults_are_documented_in_constants():
    """The defaults need to be importable for downstream code (CLI, docs)."""
    assert isinstance(DEFAULT_K, int) and DEFAULT_K > 0
    assert isinstance(DEFAULT_MIN_SCORE, float) and 0.0 < DEFAULT_MIN_SCORE < 1.0
