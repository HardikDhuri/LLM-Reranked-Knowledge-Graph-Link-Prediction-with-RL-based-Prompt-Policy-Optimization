"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def _set_api_key_env(monkeypatch):
    """Ensure AI_GATEWAY_API_KEY is set for every test and settings cache is cleared."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_triples() -> list[tuple[str, str, str]]:
    """Small synthetic KG for testing."""
    return [
        ("/m/01", "/r/capital_of", "/m/02"),
        ("/m/02", "/r/located_in", "/m/03"),
        ("/m/03", "/r/capital_of", "/m/04"),
        ("/m/01", "/r/located_in", "/m/03"),
        ("/m/04", "/r/capital_of", "/m/05"),
        ("/m/05", "/r/located_in", "/m/01"),
        ("/m/02", "/r/capital_of", "/m/03"),
        ("/m/01", "/r/capital_of", "/m/05"),
        ("/m/03", "/r/located_in", "/m/04"),
        ("/m/04", "/r/located_in", "/m/02"),
    ]


@pytest.fixture
def synthetic_entities() -> list[str]:
    return [f"/m/0{i}" for i in range(1, 6)]


@pytest.fixture
def synthetic_relations() -> list[str]:
    return ["/r/capital_of", "/r/located_in"]


# ---------------------------------------------------------------------------
# Mock LLM client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.chat_completion_json.return_value = {
        "parsed_json": {"score": 0.75, "reason": "test"},
        "content": '{"score": 0.75}',
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "latency_s": 0.5,
    }
    client.stats.return_value = {
        "total_calls": 1,
        "total_failures": 0,
        "total_tokens": 150,
        "total_latency_s": 0.5,
        "avg_latency_s": 0.5,
    }
    return client


# ---------------------------------------------------------------------------
# Mock Wikidata resolver fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_resolver() -> MagicMock:
    resolver = MagicMock()

    def mock_mid_to_text(mid: str) -> dict[str, str]:
        labels = {
            "/m/01": {
                "mid": "/m/01",
                "qid": "Q1",
                "label": "Entity One",
                "description": "First entity",
            },
            "/m/02": {
                "mid": "/m/02",
                "qid": "Q2",
                "label": "Entity Two",
                "description": "Second entity",
            },
            "/m/03": {
                "mid": "/m/03",
                "qid": "Q3",
                "label": "Entity Three",
                "description": "Third entity",
            },
            "/m/04": {
                "mid": "/m/04",
                "qid": "Q4",
                "label": "Entity Four",
                "description": "Fourth entity",
            },
            "/m/05": {
                "mid": "/m/05",
                "qid": "Q5",
                "label": "Entity Five",
                "description": "Fifth entity",
            },
        }
        return labels.get(
            mid, {"mid": mid, "qid": "Q0", "label": mid, "description": ""}
        )

    resolver.mid_to_text.side_effect = mock_mid_to_text
    return resolver


# ---------------------------------------------------------------------------
# Mock prompt template fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_template() -> MagicMock:
    template = MagicMock()
    template.id = "test_template"
    template.to_messages.return_value = [
        {"role": "system", "content": "You are a KG expert."},
        {"role": "user", "content": "Is this triple plausible?"},
    ]
    return template


# ---------------------------------------------------------------------------
# Temp results dir fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d
