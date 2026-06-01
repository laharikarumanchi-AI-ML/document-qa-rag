"""Tests for ragqa.embedding.

The Embedder is a thin wrapper over sentence-transformers. Most tests
mock SentenceTransformer so they run in milliseconds. One test marked
`@pytest.mark.slow` exercises the real model to verify the
similarity contract (similar sentences embed closer than dissimilar ones).
Run the slow test explicitly with: pytest -m slow
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ragqa.embedding import Embedder, DEFAULT_MODEL


# ───────────────────────── shape and dtype contract (mocked) ──────────────


def _patched_st(model_name: str, dim: int = 384, return_vec=None):
    """Build a MagicMock that pretends to be a SentenceTransformer."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim
    if return_vec is None:
        # Default: each text → a random unit vector of length `dim`.
        def encode_side_effect(texts, normalize_embeddings=True, **_):
            arr = np.random.RandomState(42).randn(len(texts), dim).astype(np.float32)
            if normalize_embeddings:
                arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            return arr
        mock.encode.side_effect = encode_side_effect
    else:
        mock.encode.return_value = return_vec
    return mock


def test_encode_returns_2d_array_with_correct_shape():
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL)
        emb = Embedder()
        out = emb.encode(["hello", "world", "third sentence"])
    assert out.shape == (3, 384)
    assert out.dtype == np.float32


def test_encode_empty_list_returns_empty_2d_array():
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL)
        emb = Embedder()
        out = emb.encode([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32


def test_encode_query_returns_1d_array():
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL)
        emb = Embedder()
        out = emb.encode_query("a question")
    assert out.ndim == 1
    assert out.shape == (384,)
    assert out.dtype == np.float32


def test_dim_property_reflects_model():
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL, dim=384)
        emb = Embedder()
        assert emb.dim == 384


def test_encode_normalizes_by_default():
    """Vectors must come back unit-length so cosine = dot product."""
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL)
        emb = Embedder()
        out = emb.encode(["a", "b", "c"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_model_loads_lazily():
    """Constructing an Embedder must NOT load the model — only the first
    encode call does. Keeps startup fast (sentence-transformers loads
    PyTorch + ~80MB of weights)."""
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(DEFAULT_MODEL)
        emb = Embedder()
        st.assert_not_called()  # constructing does NOT load
        emb.encode(["x"])
        st.assert_called_once_with(DEFAULT_MODEL)


def test_custom_model_name_is_passed_through():
    custom = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    with patch("ragqa.embedding.SentenceTransformer") as st:
        st.return_value = _patched_st(custom)
        emb = Embedder(model_name=custom)
        emb.encode(["x"])
        st.assert_called_once_with(custom)


# ───────────────────────── real-model integration (slow) ──────────────────


@pytest.mark.slow
def test_real_model_similar_sentences_embed_closer_than_dissimilar():
    """End-to-end with the real all-MiniLM-L6-v2 model. Run with: pytest -m slow.
    Downloads ~80MB on first run; cached afterwards."""
    emb = Embedder()
    vecs = emb.encode([
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Stock prices surged in early trading.",
    ])
    # Cosine similarity = dot product because we normalize
    sim_cat_rug = float(np.dot(vecs[0], vecs[1]))
    sim_cat_stocks = float(np.dot(vecs[0], vecs[2]))
    assert sim_cat_rug > sim_cat_stocks, (
        f"cat~rug similarity ({sim_cat_rug:.3f}) should beat cat~stocks "
        f"({sim_cat_stocks:.3f}) for a sensible embedding"
    )
