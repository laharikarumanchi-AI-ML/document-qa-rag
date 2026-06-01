"""Grounded answer generation with citation extraction.

This module is where the "every claim is grounded in a citation" promise
is structurally enforced. Two mechanisms:

1. If the retriever returned no chunks, abstain BEFORE calling the LLM.
   The model never gets a chance to fabricate from low-confidence
   material. (See ragqa.retrieval for the threshold side.)

2. The system prompt instructs the model to answer ONLY from the
   numbered context passages, citing each claim with [N] markers, OR
   to emit the exact ABSTENTION_MESSAGE if the context doesn't cover
   the question.

After generation, citation markers are parsed and mapped back to the
actual Chunk objects so the UI can highlight the cited spans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragqa.chunking import Chunk

if TYPE_CHECKING:
    from ragqa.llm import GroqClient


ABSTENTION_MESSAGE = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""\
You are a research assistant. Answer the user's question using ONLY the \
numbered context passages provided in the user message. Do not use prior \
knowledge. Do not speculate. Do not invent passage numbers.

Cite each claim by appending [N] where N is the passage number (1-indexed). \
If you use information from multiple passages, cite all of them.

If the context passages do not contain the answer to the question, respond \
with EXACTLY this sentence and nothing else:
{ABSTENTION_MESSAGE}
"""


@dataclass(frozen=True)
class Answer:
    text: str               # the model's response (or the abstention message)
    citations: list[Chunk]  # chunks cited in `text`, in order of first appearance, deduped
    abstained: bool         # True if the answer is the abstention message


def answer(
    query: str,
    *,
    retrieved: list[tuple[Chunk, float]],
    llm: "GroqClient",
) -> Answer:
    """Produce a grounded answer to `query` using the retrieved chunks.

    If `retrieved` is empty, abstain immediately without calling the LLM.
    """
    if not retrieved:
        return Answer(text=ABSTENTION_MESSAGE, citations=[], abstained=True)

    chunks = [c for c, _ in retrieved]
    messages = _build_messages(query, chunks)
    response = llm.chat(messages, temperature=0)

    # The model is instructed to emit ABSTENTION_MESSAGE exactly when the
    # context doesn't cover the question. Detect that and propagate.
    if response.strip() == ABSTENTION_MESSAGE:
        return Answer(text=ABSTENTION_MESSAGE, citations=[], abstained=True)

    indices = _parse_citations(response)
    cited_chunks: list[Chunk] = []
    for n in indices:
        if 1 <= n <= len(chunks):
            cited_chunks.append(chunks[n - 1])
        # else: silently skip — the model hallucinated a number
    return Answer(text=response, citations=cited_chunks, abstained=False)


# ───────────────────────── helpers ────────────────────────────────────────


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _parse_citations(text: str) -> list[int]:
    """Extract integer citation indices from [N] markers in `text`.

    Returns indices in order of FIRST appearance, deduplicated. So
    "[2] foo [1] bar [2]" yields [2, 1] — preserving the order the
    model used them, since that's the order a reader will encounter
    them.
    """
    seen: set[int] = set()
    out: list[int] = []
    for match in _CITATION_RE.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _build_messages(query: str, chunks: list[Chunk]) -> list[dict]:
    """Build the (system, user) messages for the LLM."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        # Header carries source + page; the actual chunk text follows.
        # The trailing blank line separates passages visually for the model.
        lines.append(f"[{i}] (source: {c.source_file}, page {c.page})")
        lines.append(c.text)
        lines.append("")
    context_block = "\n".join(lines).rstrip()

    user_content = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
