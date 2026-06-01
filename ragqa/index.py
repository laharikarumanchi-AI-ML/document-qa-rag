"""FAISS-backed similarity index over chunks.

`Index` wraps a `faiss.IndexFlatIP` (inner-product / dot-product) plus
a parallel list of the `Chunk`s that were embedded into it. Because
`Embedder` returns L2-normalized vectors, inner-product == cosine
similarity — search returns top-k chunks ranked by cosine.

On-disk layout (one directory per index):
    <path>/
        index.faiss      # binary FAISS index (faiss.write_index format)
        chunks.json      # JSON-serialized list[Chunk] in the same order
                         # as the FAISS index's internal row order
        meta.json        # {embedder_model, dim, n_chunks}
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from ragqa.chunking import Chunk

if TYPE_CHECKING:
    from ragqa.embedding import Embedder


class Index:
    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: list[Chunk],
        embedder_model: str,
        dim: int,
    ) -> None:
        self._faiss = faiss_index
        self._chunks = list(chunks)
        self._embedder_model = embedder_model
        self._dim = dim

    # ───────────────────────── construction ──────────────────────────

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: "Embedder") -> "Index":
        """Embed all chunks (one batched call) and load them into a fresh
        inner-product FAISS index. Returns even for empty input."""
        dim = embedder.dim
        faiss_index = faiss.IndexFlatIP(dim)

        if not chunks:
            return cls(faiss_index, [], embedder.model_name, dim)

        vectors = embedder.encode([c.text for c in chunks])
        # FAISS expects float32 contiguous arrays; Embedder already returns that.
        faiss_index.add(vectors)
        return cls(faiss_index, list(chunks), embedder.model_name, dim)

    # ───────────────────────── search ────────────────────────────────

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return up to k top-scoring `(Chunk, score)` pairs in
        non-increasing score order. Empty list if the index is empty."""
        if self._faiss.ntotal == 0:
            return []
        if k <= 0:
            return []
        k = min(k, self._faiss.ntotal)
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, indices = self._faiss.search(q, k)
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._chunks[int(idx)], float(score)))
        return results

    # ───────────────────────── properties ────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def embedder_model(self) -> str:
        return self._embedder_model

    @property
    def dim(self) -> int:
        return self._dim

    # ───────────────────────── persistence ───────────────────────────

    def save(self, path: Path | str) -> None:
        """Write the index + chunks + metadata to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(path / "index.faiss"))
        (path / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self._chunks], indent=2, ensure_ascii=False)
        )
        (path / "meta.json").write_text(json.dumps({
            "embedder_model": self._embedder_model,
            "dim": self._dim,
            "n_chunks": len(self._chunks),
        }, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Index":
        """Read an index written by `save`. Does not validate the
        embedder — the caller should compare `index.embedder_model`
        against their current Embedder's `model_name`."""
        path = Path(path)
        faiss_index = faiss.read_index(str(path / "index.faiss"))
        chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
        meta = json.loads((path / "meta.json").read_text())
        return cls(faiss_index, chunks, meta["embedder_model"], int(meta["dim"]))
