"""Embedding wrapper over sentence-transformers.

The Embedder loads the model lazily on first encode call. Vectors come
back float32 and unit-normalized so cosine similarity == dot product —
this matches the FAISS inner-product index used by `ragqa.index`.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper. Loads the model lazily; encodes to L2-normalized float32."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """The HF identifier of the underlying model. Saved into Index
        metadata so a query-time loader can sanity-check it matches."""
        return self._model_name

    @property
    def model(self) -> SentenceTransformer:
        """Loaded on first access. Subsequent calls reuse the same instance."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns a (len(texts), dim) float32 array. Empty input → (0, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Returns a (dim,) float32 vector for a single query."""
        return self.encode([text])[0]
