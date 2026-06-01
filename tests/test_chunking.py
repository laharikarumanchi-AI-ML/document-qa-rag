"""Tests for ragqa.chunking.

These tests are split into two layers:

- Unit tests of `_chunk_page` work on plain strings — no PDF parsing,
  fast, exhaustive of boundary edge cases.
- A couple of integration tests use `make_pdf` (conftest fixture) to
  build a real PDF on disk and exercise the full `chunk_pdf` path.
"""
from __future__ import annotations

import pytest

from ragqa.chunking import Chunk, chunk_pdf, _chunk_page, _find_boundary


# ───────────────────────── _chunk_page unit tests ──────────────────────────


def test_empty_text_produces_no_chunks():
    out = _chunk_page("", source_file="x.pdf", page=1,
                      target_chars=100, overlap_chars=20)
    assert out == []


def test_whitespace_only_text_produces_no_chunks():
    out = _chunk_page("   \n\n  \t  ", source_file="x.pdf", page=1,
                      target_chars=100, overlap_chars=20)
    assert out == []


def test_short_text_produces_single_chunk():
    text = "Hello world."
    out = _chunk_page(text, source_file="x.pdf", page=1,
                      target_chars=100, overlap_chars=20)
    assert len(out) == 1
    assert out[0].text == "Hello world."
    assert out[0].source_file == "x.pdf"
    assert out[0].page == 1
    assert out[0].char_start == 0
    assert out[0].char_end == len(text)


def test_long_text_produces_multiple_chunks():
    # 800 characters of meaningless content, no obvious boundaries
    text = "a" * 800
    out = _chunk_page(text, source_file="x.pdf", page=1,
                      target_chars=200, overlap_chars=40)
    assert len(out) >= 4  # roughly 800 / (200 - 40) chunks


def test_chunks_overlap():
    # Use distinct content per region so we can detect overlap unambiguously
    text = ("A" * 200) + ("B" * 200) + ("C" * 200)
    out = _chunk_page(text, source_file="x.pdf", page=1,
                      target_chars=200, overlap_chars=50)
    # Consecutive chunks share text near their boundary
    for i in range(len(out) - 1):
        tail_of_prev = out[i].text[-50:]
        head_of_next = out[i + 1].text[:50]
        # At least SOME char overlap (we don't insist on exact 50 because
        # boundary-snapping can shift things)
        assert any(c in head_of_next for c in tail_of_prev)


def test_offsets_correctly_reference_source():
    text = "0123456789" * 30   # 300 chars
    out = _chunk_page(text, source_file="x.pdf", page=1,
                      target_chars=100, overlap_chars=20)
    for chunk in out:
        # The text in the chunk must come from text[char_start:char_end]
        # (allowing whitespace-strip on the edges)
        original_slice = text[chunk.char_start:chunk.char_end]
        assert chunk.text == original_slice.strip()
        assert 0 <= chunk.char_start < chunk.char_end <= len(text)


def test_paragraph_boundary_preferred():
    """When a `\\n\\n` sits inside the boundary-search window, the SPLIT
    should happen there. Overlap still applies (chunk N+1 starts a few
    chars before the boundary by design — that's what overlap means)."""
    para1 = "First paragraph " * 6  # ~96 chars
    para2 = "Second paragraph " * 6
    text = para1 + "\n\n" + para2
    out = _chunk_page(text, source_file="x.pdf", page=1,
                      target_chars=110, overlap_chars=10)
    assert len(out) >= 2

    # The first chunk should end at the paragraph break, not mid-word
    # of para2.  Concretely: char_end should be right after "\n\n".
    paragraph_break_end = len(para1) + len("\n\n")
    assert out[0].char_end == paragraph_break_end, (
        f"first chunk should end at the paragraph break (idx {paragraph_break_end}); "
        f"got char_end={out[0].char_end}"
    )

    # The second chunk must contain content from para2.
    assert "Second paragraph" in out[1].text


def test_chunks_have_correct_source_file_and_page():
    out = _chunk_page("Some text on a page.", source_file="report.pdf", page=7,
                      target_chars=100, overlap_chars=20)
    assert out[0].source_file == "report.pdf"
    assert out[0].page == 7


# ───────────────────────── _find_boundary unit tests ───────────────────────


def test_find_boundary_prefers_paragraph_break():
    text = "aaa bbb\n\nccc ddd"  # "\n\n" at index 7-9
    # search window [3..12]: should return position right after "\n\n"
    pos = _find_boundary(text, start=3, end=12)
    assert pos == 9  # index just after "\n\n"


def test_find_boundary_falls_back_to_sentence():
    text = "First sentence. Second sentence. Third."
    pos = _find_boundary(text, start=5, end=20)
    assert text[pos - 2:pos] == ". "  # split came right after ". "


def test_find_boundary_falls_back_to_word():
    text = "onelongwordwithoutsentences but here is a space"
    # Window 5..30 has no \n\n or ". " so should fall to a space boundary
    pos = _find_boundary(text, start=5, end=30)
    assert text[pos - 1] == " "


def test_find_boundary_returns_end_when_no_boundary_in_window():
    text = "nospacesatallhereinthistext"
    pos = _find_boundary(text, start=5, end=20)
    assert pos == 20


# ───────────────────────── chunk_pdf integration tests ─────────────────────


def test_chunk_pdf_basic(make_pdf):
    path = make_pdf(["Hello from page one."])
    chunks = chunk_pdf(path, target_chars=600, overlap_chars=100)
    assert len(chunks) == 1
    assert "Hello from page one" in chunks[0].text
    assert chunks[0].page == 1
    assert chunks[0].source_file == path.name


def test_chunk_pdf_multi_page_chunks_dont_span_pages(make_pdf):
    """Critical: every chunk must belong to exactly one page so citations point at a single page."""
    path = make_pdf([
        "First page text. " * 30,    # ~510 chars
        "Second page text. " * 30,
        "Third page text. " * 30,
    ])
    chunks = chunk_pdf(path, target_chars=400, overlap_chars=80)

    # All chunks come from exactly one of pages 1, 2, 3
    pages_present = {c.page for c in chunks}
    assert pages_present == {1, 2, 3}

    # No chunk's text mixes content from two different pages
    for c in chunks:
        if c.page == 1:
            assert "Second page" not in c.text and "Third page" not in c.text
        elif c.page == 2:
            assert "First page" not in c.text and "Third page" not in c.text
        elif c.page == 3:
            assert "First page" not in c.text and "Second page" not in c.text


def test_chunk_pdf_empty_pages_dont_crash(make_pdf):
    """A PDF page with no text should produce zero chunks for that page (not raise)."""
    path = make_pdf(["", "Real content here."])
    chunks = chunk_pdf(path)
    # Only page 2 should have produced any chunks
    assert all(c.page == 2 for c in chunks)
    assert len(chunks) >= 1


def test_chunk_is_immutable():
    """Chunk should be a frozen dataclass — citation identities are stable."""
    c = Chunk(text="x", source_file="x.pdf", page=1, char_start=0, char_end=1)
    with pytest.raises(Exception):
        c.text = "changed"  # type: ignore[misc]
