"""Page-aware PDF chunking.

Splits a PDF into `Chunk`s where each chunk belongs to exactly one page
and carries its character offsets within that page. Two callers care
about this:

- The retriever: it embeds and searches chunk-sized pieces of text.
- The citation layer: when an answer cites chunk N, the UI maps that
  back to a specific (page, char_start, char_end) span the user can
  click through to.

Chunks never span page boundaries. Within a page, the splitter targets
`target_chars` per chunk with `overlap_chars` of overlap between
consecutive chunks, snapping to paragraph / sentence / word boundaries
when one is available in the last 20% of the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Chunk:
    """An immutable piece of source text. Identity = (source_file, page, char_start, char_end)."""
    text: str
    source_file: str   # basename of the PDF
    page: int          # 1-indexed
    char_start: int    # inclusive, offset within page
    char_end: int      # exclusive, offset within page


def chunk_pdf(
    path: Path,
    *,
    target_chars: int = 600,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Split a PDF into page-aware chunks.

    `target_chars` is approximate — the splitter snaps to nearby natural
    boundaries. `overlap_chars` is how much trailing text from one chunk
    is repeated at the start of the next, to avoid cutting relevant
    context exactly at a boundary.
    """
    path = Path(path)
    chunks: list[Chunk] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            chunks.extend(_chunk_page(
                text,
                source_file=path.name,
                page=page_num,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            ))
    return chunks


def _chunk_page(
    text: str,
    *,
    source_file: str,
    page: int,
    target_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Split a single page's text into overlapping chunks.

    Honours `target_chars` and `overlap_chars`. Skips an empty / whitespace-only
    page (returns []).
    """
    if not text or not text.strip():
        return []
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    chunks: list[Chunk] = []
    n = len(text)
    pos = 0

    while pos < n:
        nominal_end = min(pos + target_chars, n)

        if nominal_end < n:
            # Try to snap to a natural boundary in the last 20% of the window
            search_start = max(pos + int(target_chars * 0.8), pos + 1)
            boundary = _find_boundary(text, start=search_start, end=nominal_end)
            end = boundary if boundary > pos else nominal_end
        else:
            end = nominal_end

        chunk_text = text[pos:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                page=page,
                char_start=pos,
                char_end=end,
            ))

        if end >= n:
            break

        # Advance, leaving an overlap window. Always make forward progress.
        next_pos = end - overlap_chars
        if next_pos <= pos:
            next_pos = pos + 1
        pos = next_pos

    return chunks


def _find_boundary(text: str, *, start: int, end: int) -> int:
    """Return the best natural-boundary index within `text[start:end]`.

    Tries, in order: paragraph break (`\\n\\n`), sentence break (`. `, `? `,
    `! `), word boundary (` `). The returned index is the position
    AFTER the boundary marker — i.e., where the next chunk should begin.

    If no boundary is found, returns `end` (caller's fallback).
    """
    # Paragraph break — strongest signal
    idx = text.rfind("\n\n", start, end)
    if idx != -1:
        return idx + 2

    # Sentence-ending punctuation followed by a space
    for sep in (". ", "? ", "! "):
        idx = text.rfind(sep, start, end)
        if idx != -1:
            return idx + 2

    # Plain word break
    idx = text.rfind(" ", start, end)
    if idx != -1:
        return idx + 1

    # No boundary found — caller falls back to `end`
    return end
