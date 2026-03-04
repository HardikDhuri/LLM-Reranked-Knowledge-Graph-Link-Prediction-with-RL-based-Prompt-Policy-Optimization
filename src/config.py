"""Centralized configuration using pydantic-settings and python-dotenv."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Gateway
    ai_gateway_api_key: str = Field(..., description="API key for AI Gateway")
    ai_gateway_base_url: str = Field(default="https://ai-gateway.uni-paderborn.de/v1")
    ai_gateway_model: str = Field(
        default="", description="Model ID; auto-discovered if empty"
    )

    # Wikidata SPARQL
    wikidata_sparql_url: str = Field(default="https://query.wikidata.org/sparql")
    wikidata_user_agent: str = Field(default="kg-llm-rl-link-prediction/1.0")

    # Paths
    data_dir: Path = Field(default=Path("data_fb15k237"))
    cache_dir: Path = Field(default=Path("cache"))
    results_dir: Path = Field(default=Path("results"))

    # Experiment
    random_seed: int = Field(default=42)
    sample_test_queries: int = Field(default=15)
    num_candidates: int = Field(default=25)
    topk_show: int = Field(default=5)

    # RL
    rl_reward_lambda: float = Field(
        default=0.1, description="Cost penalty weight in RL reward"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def ensure_dirs(self) -> None:
        """Create data_dir, cache_dir, and results_dir if they do not exist."""
        for directory in (self.data_dir, self.cache_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


def print_config(settings: Settings | None = None) -> None:
    """Print all settings with the API key masked."""
    if settings is None:
        settings = get_settings()
    for field_name in Settings.model_fields:
        value = getattr(settings, field_name)
        if field_name == "ai_gateway_api_key":
            value = "***"
        print(f"{field_name}={value}")
