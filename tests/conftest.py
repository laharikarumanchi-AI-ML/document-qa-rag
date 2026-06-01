"""Shared pytest fixtures."""
from __future__ import annotations
from pathlib import Path
import io

import pytest


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal PDF with one page per item in `pages`.

    Each page renders the given string starting near the top-left.
    Long pages get word-wrapped naively. The result is a real PDF
    parseable by pdfplumber.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    page_width, page_height = LETTER
    margin = 72
    line_height = 14
    max_line_chars = 90  # rough wrap

    for page_text in pages:
        # Naive word-wrap so long pages render multiple lines
        lines: list[str] = []
        for paragraph in page_text.split("\n"):
            current = ""
            for word in paragraph.split(" "):
                if len(current) + len(word) + 1 > max_line_chars and current:
                    lines.append(current)
                    current = word
                else:
                    current = (current + " " + word).strip()
            if current:
                lines.append(current)
            lines.append("")  # paragraph break

        y = page_height - margin
        for line in lines:
            if y < margin:
                break
            c.drawString(margin, y, line)
            y -= line_height
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Factory fixture: call make_pdf(['page1 text', 'page2 text']) → returns Path."""
    counter = {"n": 0}

    def _create(pages: list[str]) -> Path:
        counter["n"] += 1
        path = tmp_path / f"fixture-{counter['n']}.pdf"
        path.write_bytes(_make_pdf_bytes(pages))
        return path

    return _create
