"""Tests for ragqa.answer.

The answer module is where the "every claim is grounded in a citation"
promise gets enforced. Two structural mechanisms:

1. If retrieval is empty, abstain BEFORE calling the LLM. The model
   never gets a chance to make something up.
2. The system prompt forces the LLM to either cite [N] for every
   claim or emit the exact ABSTENTION_MESSAGE.

These tests use a mocked LLM (no API calls) and verify the prompt
shape, the citation parser, and the abstention plumbing.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragqa.chunking import Chunk
from ragqa.answer import (
    Answer,
    answer,
    ABSTENTION_MESSAGE,
    _parse_citations,
    _build_messages,
)


def _chunk(text: str, page: int = 1, source: str = "x.pdf") -> Chunk:
    return Chunk(text=text, source_file=source, page=page,
                 char_start=0, char_end=len(text))


def _retrieved(chunks_and_scores: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    return chunks_and_scores


def _fake_llm(response: str):
    llm = MagicMock()
    llm.chat = MagicMock(return_value=response)
    return llm


# ───────────────────────── abstention ──────────────────────────────────────


def test_empty_retrieval_returns_abstention_without_calling_llm():
    """If the retriever returned nothing, we never even ask the LLM —
    that's the structural anti-hallucination claim."""
    llm = _fake_llm("(should never be called)")
    out = answer("does this say X?", retrieved=[], llm=llm)
    assert out.abstained is True
    assert out.text == ABSTENTION_MESSAGE
    assert out.citations == []
    llm.chat.assert_not_called()


def test_model_can_also_abstain_explicitly():
    """If the model itself returns the abstention message, propagate that
    as abstained=True (with no citations)."""
    llm = _fake_llm(ABSTENTION_MESSAGE)
    out = answer("Q?", retrieved=[(_chunk("a"), 0.9)], llm=llm)
    assert out.abstained is True
    assert out.text == ABSTENTION_MESSAGE
    assert out.citations == []


# ───────────────────────── answer happy path ───────────────────────────────


def test_returns_model_response_when_citations_present():
    chunks = [_chunk("first passage"), _chunk("second passage")]
    llm = _fake_llm("The answer involves [1] and also [2].")
    out = answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    assert out.abstained is False
    assert "The answer involves" in out.text
    assert len(out.citations) == 2


def test_citations_map_back_to_correct_chunks():
    chunks = [_chunk("alpha"), _chunk("beta"), _chunk("gamma")]
    llm = _fake_llm("It says alpha [1] and gamma [3].")
    out = answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    assert [c.text for c in out.citations] == ["alpha", "gamma"]


def test_citations_dedup_and_preserve_first_appearance_order():
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    llm = _fake_llm("[2] and then [1] and again [2] and [3].")
    out = answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    # Indices appear: 2, 1, 2, 3. Deduped + first-appearance order: 2, 1, 3.
    assert [c.text for c in out.citations] == ["b", "a", "c"]


def test_invalid_citation_indices_are_silently_skipped():
    """If the model hallucinates [99] when only 2 chunks exist, drop it
    rather than crash. The remaining valid citations still go through."""
    chunks = [_chunk("a"), _chunk("b")]
    llm = _fake_llm("First, [1] confirms it. Also [99] which doesn't exist.")
    out = answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    assert [c.text for c in out.citations] == ["a"]


def test_response_with_no_citations_still_returned():
    """The model not citing is technically a prompt failure, but we don't
    drop the response — we return it with citations=[] so the caller
    can decide whether to display a warning."""
    chunks = [_chunk("alpha")]
    llm = _fake_llm("The answer is yes.")
    out = answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    assert out.abstained is False
    assert out.text == "The answer is yes."
    assert out.citations == []


# ───────────────────────── _parse_citations unit tests ────────────────────


def test_parse_citations_extracts_in_order():
    assert _parse_citations("[2] first, then [1].") == [2, 1]


def test_parse_citations_dedupes():
    assert _parse_citations("[1] and [1] and [2].") == [1, 2]


def test_parse_citations_handles_no_markers():
    assert _parse_citations("No citations here.") == []


def test_parse_citations_ignores_unrelated_brackets():
    """[abc] or [1.5] shouldn't match — only integer citations."""
    assert _parse_citations("This is [abc] not [1.5] but [3] yes.") == [3]


# ───────────────────────── _build_messages ─────────────────────────────────


def test_build_messages_includes_numbered_passages():
    chunks = [_chunk("first"), _chunk("second")]
    msgs = _build_messages("the question?", chunks)
    # The "user" message (last one) should contain numbered context
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert "[1]" in user_msg["content"]
    assert "[2]" in user_msg["content"]
    assert "first" in user_msg["content"]
    assert "second" in user_msg["content"]


def test_build_messages_includes_query():
    msgs = _build_messages("what is X?", [_chunk("ctx")])
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert "what is X?" in user_msg["content"]


def test_build_messages_system_prompt_mentions_abstention_message_exactly():
    """The model needs to know the EXACT abstention string we'll detect."""
    msgs = _build_messages("Q?", [_chunk("ctx")])
    system_msg = next(m for m in msgs if m["role"] == "system")
    assert ABSTENTION_MESSAGE in system_msg["content"]


def test_build_messages_system_prompt_forbids_outside_knowledge():
    """Cheap text-pattern check that the prompt explicitly constrains the model."""
    msgs = _build_messages("Q?", [_chunk("ctx")])
    system_msg = next(m for m in msgs if m["role"] == "system")
    # Some words that signal grounding constraints. Tolerant of rewording.
    content_lower = system_msg["content"].lower()
    assert "only" in content_lower
    assert "context" in content_lower or "passages" in content_lower


# ───────────────────────── temperature default ─────────────────────────────


def test_temperature_zero_passed_by_default():
    """Answer generation should be reproducible — pin temperature=0 by default."""
    chunks = [_chunk("a")]
    llm = _fake_llm("ok [1].")
    answer("Q?", retrieved=[(c, 0.9) for c in chunks], llm=llm)
    _, kwargs = llm.chat.call_args
    assert kwargs.get("temperature") == 0


# ───────────────────────── Answer is immutable ─────────────────────────────


def test_answer_is_frozen():
    a = Answer(text="x", citations=[], abstained=False)
    with pytest.raises(Exception):
        a.text = "changed"  # type: ignore[misc]
