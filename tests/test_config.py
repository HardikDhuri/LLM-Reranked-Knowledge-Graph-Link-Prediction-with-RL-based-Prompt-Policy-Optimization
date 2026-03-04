"""Tests for src/config.py."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings, print_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_ENV = {"AI_GATEWAY_API_KEY": "test-key"}


# ---------------------------------------------------------------------------
# Settings creation
# ---------------------------------------------------------------------------


def test_settings_created_with_env_vars(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "my-key")
    settings = Settings()
    assert settings.ai_gateway_api_key == "my-key"


def test_missing_api_key_raises_validation_error(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_values(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "key")
    settings = Settings()
    assert settings.ai_gateway_base_url == "https://ai-gateway.uni-paderborn.de/v1"
    assert settings.ai_gateway_model == ""
    assert settings.wikidata_sparql_url == "https://query.wikidata.org/sparql"
    assert settings.wikidata_user_agent == "kg-llm-rl-link-prediction/1.0"
    assert settings.data_dir == Path("data_fb15k237")
    assert settings.cache_dir == Path("cache")
    assert settings.results_dir == Path("results")
    assert settings.random_seed == 42
    assert settings.sample_test_queries == 15
    assert settings.num_candidates == 25
    assert settings.topk_show == 5
    assert settings.rl_reward_lambda == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# print_config masks the API key
# ---------------------------------------------------------------------------


def test_print_config_masks_api_key(monkeypatch, capsys):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "super-secret")
    settings = Settings()
    print_config(settings)
    captured = capsys.readouterr()
    assert "super-secret" not in captured.out
    assert "ai_gateway_api_key=***" in captured.out


# ---------------------------------------------------------------------------
# ensure_dirs creates directories
# ---------------------------------------------------------------------------


def test_ensure_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "key")
    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
    )
    settings.ensure_dirs()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "cache").is_dir()
    assert (tmp_path / "results").is_dir()


# ---------------------------------------------------------------------------
# get_settings singleton
# ---------------------------------------------------------------------------


def test_get_settings_returns_same_instance(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "key")
    # Clear lru_cache so this test is self-contained.
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
