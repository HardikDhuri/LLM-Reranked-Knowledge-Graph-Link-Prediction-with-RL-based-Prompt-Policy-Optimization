"""Unit tests for src/models/llm_client.py."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.models.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY = "test-api-key"
_BASE_URL = "https://fake-gateway.example.com/v1"
_MODEL = "test-model"


def _make_client(**kwargs) -> LLMClient:
    """Return an LLMClient with fake credentials (no real network)."""
    defaults = {"api_key": _API_KEY, "base_url": _BASE_URL, "model": _MODEL}
    defaults.update(kwargs)
    with patch("src.models.llm_client.get_settings"):
        return LLMClient(**defaults)


def _completion_response(content: str, total_tokens: int = 10) -> MagicMock:
    """Build a mock requests.Response for a successful chat completion."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": total_tokens},
    }
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_uses_provided_values():
    client = _make_client()
    assert client.api_key == _API_KEY
    assert client.base_url == _BASE_URL.rstrip("/")
    assert client.model == _MODEL


def test_constructor_strips_trailing_slash():
    client = _make_client(base_url="https://example.com/v1/")
    assert not client.base_url.endswith("/")


def test_constructor_initial_stats_are_zero():
    client = _make_client()
    assert client.total_calls == 0
    assert client.total_failures == 0
    assert client.total_tokens == 0
    assert client.total_latency == 0.0


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def test_list_models_returns_sorted_ids():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-c"}]
    }
    client._session.get = MagicMock(return_value=mock_resp)

    models = client.list_models()

    assert models == ["model-a", "model-b", "model-c"]
    client._session.get.assert_called_once_with(f"{_BASE_URL}/models", timeout=15)


def test_list_models_returns_empty_on_error():
    client = _make_client()
    client._session.get = MagicMock(side_effect=requests.ConnectionError("down"))

    models = client.list_models()

    assert models == []


def test_list_models_returns_empty_when_data_missing():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {}
    client._session.get = MagicMock(return_value=mock_resp)

    models = client.list_models()

    assert models == []


# ---------------------------------------------------------------------------
# chat_completion – happy path
# ---------------------------------------------------------------------------


def test_chat_completion_returns_expected_keys():
    client = _make_client()
    client._session.post = MagicMock(return_value=_completion_response("hello"))

    result = client.chat_completion([{"role": "user", "content": "hi"}])

    assert result["content"] == "hello"
    assert "usage" in result
    assert "latency_s" in result
    assert "raw" in result


def test_chat_completion_updates_stats():
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response("ok", total_tokens=42)
    )

    client.chat_completion([{"role": "user", "content": "ping"}])

    assert client.total_calls == 1
    assert client.total_tokens == 42
    assert client.total_latency > 0


def test_chat_completion_sends_response_format_when_provided():
    client = _make_client()
    client._session.post = MagicMock(return_value=_completion_response("{}"))

    client.chat_completion(
        [{"role": "user", "content": "x"}],
        response_format={"type": "json_object"},
    )

    payload = client._session.post.call_args.kwargs["json"]
    assert payload["response_format"] == {"type": "json_object"}


def test_chat_completion_omits_response_format_when_none():
    client = _make_client()
    client._session.post = MagicMock(return_value=_completion_response("x"))

    client.chat_completion([{"role": "user", "content": "x"}])

    payload = client._session.post.call_args.kwargs["json"]
    assert "response_format" not in payload


# ---------------------------------------------------------------------------
# chat_completion – retry logic
# ---------------------------------------------------------------------------


def test_chat_completion_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client()

    rate_limit_resp = MagicMock()
    rate_limit_resp.status_code = 429
    rate_limit_resp.raise_for_status = MagicMock()

    ok_resp = _completion_response("done")

    client._session.post = MagicMock(side_effect=[rate_limit_resp, ok_resp])

    result = client.chat_completion([{"role": "user", "content": "x"}])

    assert result["content"] == "done"
    assert client._session.post.call_count == 2


def test_chat_completion_raises_after_three_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client()
    client._session.post = MagicMock(
        side_effect=requests.ConnectionError("network error")
    )

    with pytest.raises(requests.ConnectionError):
        client.chat_completion([{"role": "user", "content": "x"}])

    assert client._session.post.call_count == 3
    assert client.total_failures == 3


# ---------------------------------------------------------------------------
# chat_completion_json
# ---------------------------------------------------------------------------


def test_chat_completion_json_parses_valid_json():
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response('{"score": 0.95}')
    )

    result = client.chat_completion_json([{"role": "user", "content": "rate it"}])

    assert result["parsed_json"] == {"score": 0.95}
    assert result["content"] == '{"score": 0.95}'
    assert "usage" in result
    assert "latency_s" in result


def test_chat_completion_json_falls_back_to_extract_json():
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response('Sure! Here is the answer: {"score": 0.8}')
    )

    result = client.chat_completion_json([{"role": "user", "content": "rate it"}])

    assert result["parsed_json"] == {"score": 0.8}


def test_chat_completion_json_returns_none_when_unparseable():
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response("No JSON here at all.")
    )

    result = client.chat_completion_json([{"role": "user", "content": "rate it"}])

    assert result["parsed_json"] is None


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"key": "val"}', {"key": "val"}),
        ('prefix {"key": "val"} suffix', {"key": "val"}),
        ('```json\n{"key": "val"}\n```', {"key": "val"}),
        ('```\n{"key": "val"}\n```', {"key": "val"}),
        ("no json here", None),
        ("", None),
    ],
)
def test_extract_json(text, expected):
    assert LLMClient._extract_json(text) == expected


# ---------------------------------------------------------------------------
# stats & reset_stats
# ---------------------------------------------------------------------------


def test_stats_returns_correct_values(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response("ok", total_tokens=5)
    )
    client.chat_completion([{"role": "user", "content": "x"}])

    s = client.stats()
    assert s["total_calls"] == 1
    assert s["total_tokens"] == 5
    assert s["total_failures"] == 0
    assert s["total_latency_s"] >= 0
    assert s["avg_latency_s"] >= 0


def test_reset_stats_clears_counters(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client()
    client._session.post = MagicMock(return_value=_completion_response("ok"))
    client.chat_completion([{"role": "user", "content": "x"}])
    assert client.total_calls == 1

    client.reset_stats()

    assert client.total_calls == 0
    assert client.total_failures == 0
    assert client.total_tokens == 0
    assert client.total_latency == 0.0
    s = client.stats()
    assert s["avg_latency_s"] == 0.0


def test_stats_avg_latency_safe_when_no_calls():
    client = _make_client()
    s = client.stats()
    assert s["avg_latency_s"] == 0.0


# ---------------------------------------------------------------------------
# Authorization header
# ---------------------------------------------------------------------------


def test_session_has_authorization_header():
    client = _make_client()
    assert client._session.headers["Authorization"] == f"Bearer {_API_KEY}"
    assert client._session.headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Integration: stats after json completion
# ---------------------------------------------------------------------------


def test_chat_completion_json_records_stats():
    client = _make_client()
    client._session.post = MagicMock(
        return_value=_completion_response('{"score": 1}', total_tokens=20)
    )
    client.chat_completion_json([{"role": "user", "content": "x"}])

    assert client.total_calls == 1
    assert client.total_tokens == 20


# ---------------------------------------------------------------------------
# Serialization of payload fields
# ---------------------------------------------------------------------------


def test_chat_completion_payload_fields():
    client = _make_client()
    client._session.post = MagicMock(return_value=_completion_response("x"))

    client.chat_completion(
        [{"role": "user", "content": "hello"}],
        temperature=0.5,
        max_tokens=128,
    )

    payload = client._session.post.call_args.kwargs["json"]
    assert payload["model"] == _MODEL
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 128
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
