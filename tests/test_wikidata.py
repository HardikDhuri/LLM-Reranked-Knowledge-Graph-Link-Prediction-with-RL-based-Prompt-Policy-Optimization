"""Tests for src/wikidata/cache.py and src/wikidata/sparql.py."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from src.wikidata.cache import JSONCache
from src.wikidata.sparql import MIN_REQUEST_INTERVAL, WikidataResolver, _rate_limit

# ---------------------------------------------------------------------------
# JSONCache tests
# ---------------------------------------------------------------------------


def test_json_cache_set_get_has(tmp_path):
    cache = JSONCache(tmp_path / "test.json")
    assert not cache.has("key1")
    cache.set("key1", "value1")
    assert cache.has("key1")
    assert cache.get("key1") == "value1"


def test_json_cache_get_missing_returns_none(tmp_path):
    cache = JSONCache(tmp_path / "test.json")
    assert cache.get("missing") is None


def test_json_cache_contains(tmp_path):
    cache = JSONCache(tmp_path / "test.json")
    cache.set("x", 42)
    assert "x" in cache
    assert "y" not in cache


def test_json_cache_len(tmp_path):
    cache = JSONCache(tmp_path / "test.json")
    assert len(cache) == 0
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2


def test_json_cache_save_and_reload(tmp_path):
    filepath = tmp_path / "subdir" / "cache.json"
    cache = JSONCache(filepath)
    cache.set("hello", {"nested": True})
    cache.save()

    assert filepath.exists()
    reloaded = JSONCache(filepath)
    assert reloaded.get("hello") == {"nested": True}
    assert len(reloaded) == 1


def test_json_cache_save_creates_parent_dirs(tmp_path):
    filepath = tmp_path / "a" / "b" / "c" / "data.json"
    cache = JSONCache(filepath)
    cache.set("k", "v")
    cache.save()
    assert filepath.exists()
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    assert data == {"k": "v"}


def test_json_cache_loads_existing_file(tmp_path):
    filepath = tmp_path / "existing.json"
    filepath.write_text(json.dumps({"pre": "loaded"}), encoding="utf-8")
    cache = JSONCache(filepath)
    assert cache.get("pre") == "loaded"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MID_RESPONSE = {
    "results": {
        "bindings": [{"item": {"value": "http://www.wikidata.org/entity/Q76"}}]
    }
}

_TEXT_RESPONSE = {
    "results": {
        "bindings": [
            {
                "label": {"value": "Barack Obama"},
                "description": {"value": "44th president of the United States"},
            }
        ]
    }
}

_EMPTY_RESPONSE: dict = {"results": {"bindings": []}}


@pytest.fixture()
def resolver(tmp_path, monkeypatch):
    """Return a WikidataResolver backed by a temp cache dir."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    from src.config import get_settings

    get_settings.cache_clear()
    return WikidataResolver(cache_dir=tmp_path / "cache")


# ---------------------------------------------------------------------------
# WikidataResolver.mid_to_qid
# ---------------------------------------------------------------------------


def test_mid_to_qid_returns_qid(resolver):
    with patch("src.wikidata.sparql._sparql_query", return_value=_MID_RESPONSE):
        qid = resolver.mid_to_qid("/m/02mjmr")
    assert qid == "Q76"


def test_mid_to_qid_no_results_returns_none(resolver):
    with patch("src.wikidata.sparql._sparql_query", return_value=_EMPTY_RESPONSE):
        qid = resolver.mid_to_qid("/m/unknown")
    assert qid is None


def test_mid_to_qid_caches_result(resolver):
    with patch(
        "src.wikidata.sparql._sparql_query", return_value=_MID_RESPONSE
    ) as mock_q:
        resolver.mid_to_qid("/m/02mjmr")
        resolver.mid_to_qid("/m/02mjmr")  # second call should hit cache
    assert mock_q.call_count == 1


def test_mid_to_qid_caches_none(resolver):
    with patch(
        "src.wikidata.sparql._sparql_query", return_value=_EMPTY_RESPONSE
    ) as mock_q:
        resolver.mid_to_qid("/m/none")
        resolver.mid_to_qid("/m/none")
    assert mock_q.call_count == 1


# ---------------------------------------------------------------------------
# WikidataResolver.get_entity_text
# ---------------------------------------------------------------------------


def test_get_entity_text_returns_label_and_description(resolver):
    with patch("src.wikidata.sparql._sparql_query", return_value=_TEXT_RESPONSE):
        text = resolver.get_entity_text("Q76")
    assert text["label"] == "Barack Obama"
    assert text["description"] == "44th president of the United States"


def test_get_entity_text_empty_falls_back_to_qid(resolver):
    with patch("src.wikidata.sparql._sparql_query", return_value=_EMPTY_RESPONSE):
        text = resolver.get_entity_text("Q999")
    assert text["label"] == "Q999"
    assert text["description"] == ""


def test_get_entity_text_caches_result(resolver):
    with patch(
        "src.wikidata.sparql._sparql_query", return_value=_TEXT_RESPONSE
    ) as mock_q:
        resolver.get_entity_text("Q76")
        resolver.get_entity_text("Q76")
    assert mock_q.call_count == 1


# ---------------------------------------------------------------------------
# WikidataResolver.mid_to_text
# ---------------------------------------------------------------------------


def test_mid_to_text_full_pipeline(resolver):
    def fake_sparql(query, **_):
        if "P646" in query:
            return _MID_RESPONSE
        return _TEXT_RESPONSE

    with patch("src.wikidata.sparql._sparql_query", side_effect=fake_sparql):
        result = resolver.mid_to_text("/m/02mjmr")

    assert result["mid"] == "/m/02mjmr"
    assert result["qid"] == "Q76"
    assert result["label"] == "Barack Obama"
    assert result["description"] == "44th president of the United States"


def test_mid_to_text_fallback_when_no_qid(resolver):
    with patch("src.wikidata.sparql._sparql_query", return_value=_EMPTY_RESPONSE):
        result = resolver.mid_to_text("/m/unknown")
    assert result["mid"] == "/m/unknown"
    assert result["qid"] is None
    assert result["label"] == "/m/unknown"
    assert result["description"] == ""


# ---------------------------------------------------------------------------
# WikidataResolver.cache_stats
# ---------------------------------------------------------------------------


def test_cache_stats(resolver):
    stats = resolver.cache_stats()
    assert stats["mid2qid_cached"] == 0
    assert stats["entity_text_cached"] == 0

    with patch("src.wikidata.sparql._sparql_query", return_value=_MID_RESPONSE):
        resolver.mid_to_qid("/m/02mjmr")
    stats = resolver.cache_stats()
    assert stats["mid2qid_cached"] == 1


# ---------------------------------------------------------------------------
# Rate limiting (optional)
# ---------------------------------------------------------------------------


def test_rate_limit_sleeps_when_called_quickly():
    """_rate_limit should sleep when called within MIN_REQUEST_INTERVAL."""
    import src.wikidata.sparql as sparql_mod

    # Set _last_request_time to just now so the next call must sleep.
    sparql_mod._last_request_time = time.time()
    with patch("src.wikidata.sparql.time.sleep") as mock_sleep:
        _rate_limit()
    assert mock_sleep.called
    sleep_arg = mock_sleep.call_args[0][0]
    assert 0 < sleep_arg <= MIN_REQUEST_INTERVAL
