"""Top-k chunk retrieval with an abstention threshold.

`retrieve()` is the single function the answer module calls. It:

1. Encodes the query with the same embedder family used at index time.
2. Asks the index for the top-k nearest chunks by cosine similarity.
3. Filters out chunks whose score is below `min_score`.

If the returned list is empty, the answer module abstains. The
threshold is the *structural* anti-hallucination mechanism: the LLM
never sees low-confidence material, so it can't make something up
from it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ragqa.chunking import Chunk

if TYPE_CHECKING:
    from ragqa.embedding import Embedder
    from ragqa.index import Index


# Defaults are module-level constants so the CLI and docs can import
# them without instantiating anything.
DEFAULT_K: int = 5
DEFAULT_MIN_SCORE: float = 0.35   # tuned to the all-MiniLM family;
                                  # tweak per-corpus if recall feels low


def retrieve(
    query: str,
    *,
    index: "Index",
    embedder: "Embedder",
    k: int = DEFAULT_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[tuple[Chunk, float]]:
    """Return the up-to-k chunks above `min_score` for a query.

    Raises ValueError if the embedder used here is not the same model
    family as the one the index was built with — silent model swaps
    tank retrieval quality without an observable failure otherwise.
    """
    if index.embedder_model != embedder.model_name:
        raise ValueError(
            f"Embedder mismatch: index built with model {index.embedder_model!r}, "
            f"but query embedder uses {embedder.model_name!r}. "
            f"Either rebuild the index or query with the matching embedder."
        )

    query_vec = embedder.encode_query(query)
    candidates = index.search(query_vec, k=k)
    above = [(chunk, score) for chunk, score in candidates if score >= min_score]
    # Belt-and-suspenders cap: index.search should already return ≤k, but
    # if anyone wires up a search that returns more, honour our contract.
    return above[:k]
