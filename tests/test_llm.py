"""Tests for ragqa.llm.GroqClient.

Same Retry-After-aware backoff pattern as superpowers/agent/llm_client.py
— battle-tested there and replicated here so the answer module has a
trustworthy HTTP layer without depending on superpowers.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import requests

from ragqa.llm import GroqClient


def test_returns_message_content_on_success():
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, world."}}]
    }
    with patch("ragqa.llm.requests.post", return_value=mock_response) as mock_post:
        client = GroqClient(api_key="k", model="llama-3.3-70b-versatile")
        out = client.chat([{"role": "user", "content": "hi"}])
    assert out == "Hello, world."

    args, kwargs = mock_post.call_args
    assert "groq.com" in args[0]
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert kwargs["json"]["model"] == "llama-3.3-70b-versatile"


def test_passes_extra_kwargs_through_to_payload():
    """temperature, max_tokens, etc. should be forwarded as part of the
    request payload — used at eval time for reproducibility."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    with patch("ragqa.llm.requests.post", return_value=mock_response) as mock_post:
        client = GroqClient(api_key="k", model="m")
        client.chat([{"role": "user", "content": "x"}], temperature=0, max_tokens=100)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 100


def test_retries_on_rate_limit_then_succeeds():
    """429 → backoff → retry → success. Default backoff (no Retry-After header)."""
    bad = MagicMock(status_code=429)
    bad.headers = {}
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    with patch("ragqa.llm.requests.post", side_effect=[bad, bad, good]):
        with patch("ragqa.llm.time.sleep") as sleep:
            client = GroqClient(api_key="k", model="m")
            out = client.chat([{"role": "user", "content": "x"}])
    assert out == "ok"
    assert sleep.call_count == 2  # two backoffs between three attempts


def test_respects_retry_after_header():
    """When Groq returns a Retry-After header, sleep for that long
    (capped at MAX_BACKOFF_SECONDS)."""
    bad = MagicMock(status_code=429)
    bad.headers = {"Retry-After": "7"}
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    with patch("ragqa.llm.requests.post", side_effect=[bad, good]):
        with patch("ragqa.llm.time.sleep") as sleep:
            GroqClient(api_key="k", model="m").chat(
                [{"role": "user", "content": "x"}]
            )
    sleep.assert_called_once_with(7.0)


def test_retry_after_is_capped_at_max_backoff():
    bad = MagicMock(status_code=429)
    bad.headers = {"Retry-After": "9999"}
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    good = MagicMock(status_code=200)
    good.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    with patch("ragqa.llm.requests.post", side_effect=[bad, good]):
        with patch("ragqa.llm.time.sleep") as sleep:
            GroqClient(api_key="k", model="m").chat(
                [{"role": "user", "content": "x"}]
            )
    sleep.assert_called_once_with(GroqClient.MAX_BACKOFF_SECONDS)


def test_attempts_up_to_max_then_raises():
    """All attempts fail with 429 → propagate HTTPError after final sleep."""
    bad = MagicMock(status_code=429)
    bad.headers = {}
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    with patch("ragqa.llm.requests.post", return_value=bad):
        with patch("ragqa.llm.time.sleep") as sleep:
            client = GroqClient(api_key="k", model="m")
            with pytest.raises(requests.HTTPError):
                client.chat([{"role": "user", "content": "x"}])
    assert sleep.call_count == GroqClient.MAX_ATTEMPTS - 1  # N-1 sleeps for N attempts


def test_non_retryable_4xx_raises_immediately():
    """A 401 (bad key) shouldn't retry — wrong key won't fix itself."""
    bad = MagicMock(status_code=401)
    bad.headers = {}
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)
    with patch("ragqa.llm.requests.post", return_value=bad):
        with patch("ragqa.llm.time.sleep") as sleep:
            with pytest.raises(requests.HTTPError):
                GroqClient(api_key="k", model="m").chat(
                    [{"role": "user", "content": "x"}]
                )
    sleep.assert_not_called()  # no retries for non-retryable status
