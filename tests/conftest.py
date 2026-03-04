"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def _set_api_key_env(monkeypatch):
    """Ensure AI_GATEWAY_API_KEY is set for every test and settings cache is cleared."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
