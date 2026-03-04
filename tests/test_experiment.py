"""Unit tests for src/experiment.py.

All external dependencies (LLM, Wikidata, dataset download, PyKEEN training)
are fully mocked so tests run without internet access or real data.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.eval.metrics import RankingResult
from src.experiment import ExperimentRunner

# ---------------------------------------------------------------------------
# Tiny synthetic dataset helpers
# ---------------------------------------------------------------------------

_TRAIN = [
    ("e0", "r0", "e1"),
    ("e1", "r0", "e2"),
    ("e2", "r0", "e3"),
    ("e3", "r0", "e4"),
    ("e4", "r0", "e0"),
    ("e0", "r1", "e2"),
    ("e1", "r1", "e3"),
    ("e2", "r1", "e4"),
]

_VALID = [
    ("e0", "r0", "e3"),
    ("e1", "r1", "e0"),
]

_TEST = [
    ("e0", "r0", "e2"),
    ("e1", "r0", "e3"),
    ("e2", "r0", "e4"),
]


def _make_dataset():
    """Return a mocked FB15k237Dataset with tiny synthetic data."""
    from src.data.fb15k237 import FB15k237Dataset

    return FB15k237Dataset(train=_TRAIN, valid=_VALID, test=_TEST)


def _make_ranking_result(query):
    h, r, t = query
    return RankingResult(
        query=query,
        true_rank=1,
        num_candidates=5,
        scored_candidates=[(t, 0.9), ("e_other", 0.1)],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_dataset():
    return _make_dataset()


@pytest.fixture()
def tmp_results(tmp_path, monkeypatch):
    """Redirect results_dir to a tmp_path and clear settings cache."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    from src.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_default_values():
    """ExperimentRunner stores defaults from settings when no overrides given."""
    from src.config import get_settings

    settings = get_settings()
    runner = ExperimentRunner()
    assert runner.template_id == "minimal"
    assert runner.embedding_epochs == 50
    assert runner.embedding_dim == 128
    assert runner.num_test_queries == settings.sample_test_queries
    assert runner.num_candidates == settings.num_candidates
    assert runner.random_seed == settings.random_seed


def test_init_custom_values():
    """Custom constructor arguments override defaults."""
    runner = ExperimentRunner(
        template_id="chain_of_thought",
        embedding_epochs=10,
        embedding_dim=64,
        num_test_queries=5,
        num_candidates=20,
        random_seed=99,
    )
    assert runner.template_id == "chain_of_thought"
    assert runner.embedding_epochs == 10
    assert runner.embedding_dim == 64
    assert runner.num_test_queries == 5
    assert runner.num_candidates == 20
    assert runner.random_seed == 99


def test_init_components_none_before_run():
    """All pipeline components should be None before run() is called."""
    runner = ExperimentRunner()
    assert runner.dataset is None
    assert runner.embedding is None
    assert runner.reranker is None


# ---------------------------------------------------------------------------
# run() tests (fully mocked)
# ---------------------------------------------------------------------------


@patch("src.experiment.WikidataResolver")
@patch("src.experiment.LLMClient")
@patch("src.experiment.PromptManager")
@patch("src.experiment.EmbeddingBaseline")
@patch("src.experiment.load_fb15k237")
def test_run_returns_dict_with_expected_keys(
    mock_load,
    mock_embedding_cls,
    mock_pm_cls,
    mock_llm_cls,
    mock_resolver_cls,
    tmp_results,
):
    dataset = _make_dataset()
    mock_load.return_value = dataset

    # Mock embedding baseline
    mock_embedding = MagicMock()
    mock_embedding.train.return_value = {
        "model": "RotatE",
        "embedding_dim": 16,
        "num_epochs": 1,
        "num_entities": 5,
        "num_relations": 2,
        "num_train_triples": len(_TRAIN),
        "device": "cpu",
    }
    mock_embedding.get_score_fn.return_value = lambda h, r, t: 0.5
    mock_embedding_cls.return_value = mock_embedding

    # Mock LLM client
    mock_llm = MagicMock()
    mock_llm.chat_completion_json.return_value = {
        "parsed_json": {"score": 0.5},
        "content": '{"score": 0.5}',
        "usage": {},
        "latency_s": 0.01,
    }
    mock_llm.stats.return_value = {
        "total_calls": 0,
        "total_failures": 0,
        "total_tokens": 0,
        "total_latency_s": 0.0,
        "avg_latency_s": 0.0,
    }
    mock_llm_cls.return_value = mock_llm

    # Mock WikidataResolver
    mock_resolver = MagicMock()
    mock_resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": f"desc of {mid}",
    }
    mock_resolver_cls.return_value = mock_resolver

    # Mock PromptManager / template
    mock_template = MagicMock()
    mock_template.id = "minimal"
    mock_template.render.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    mock_pm = MagicMock()
    mock_pm.get.return_value = mock_template
    mock_pm_cls.return_value = mock_pm

    runner = ExperimentRunner(
        template_id="minimal",
        embedding_epochs=1,
        embedding_dim=16,
        num_test_queries=2,
        num_candidates=5,
        random_seed=42,
    )
    log = runner.run()

    assert isinstance(log, dict)
    for key in [
        "timestamp",
        "config",
        "dataset_summary",
        "embedding_training",
        "embedding_metrics",
        "llm_metrics",
        "llm_reranker_stats",
        "elapsed_s",
    ]:
        assert key in log, f"Missing key: {key}"


@patch("src.experiment.WikidataResolver")
@patch("src.experiment.LLMClient")
@patch("src.experiment.PromptManager")
@patch("src.experiment.EmbeddingBaseline")
@patch("src.experiment.load_fb15k237")
def test_run_saves_results_file(
    mock_load,
    mock_embedding_cls,
    mock_pm_cls,
    mock_llm_cls,
    mock_resolver_cls,
    tmp_results,
):
    dataset = _make_dataset()
    mock_load.return_value = dataset

    mock_embedding = MagicMock()
    mock_embedding.train.return_value = {"model": "RotatE", "num_entities": 5}
    mock_embedding.get_score_fn.return_value = lambda h, r, t: 0.5
    mock_embedding_cls.return_value = mock_embedding

    mock_llm = MagicMock()
    mock_llm.chat_completion_json.return_value = {
        "parsed_json": {"score": 0.5},
        "content": '{"score": 0.5}',
        "usage": {},
        "latency_s": 0.01,
    }
    mock_llm.stats.return_value = {
        "total_calls": 0,
        "total_failures": 0,
        "total_tokens": 0,
        "total_latency_s": 0.0,
        "avg_latency_s": 0.0,
    }
    mock_llm_cls.return_value = mock_llm

    mock_resolver = MagicMock()
    mock_resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": "",
    }
    mock_resolver_cls.return_value = mock_resolver

    mock_template = MagicMock()
    mock_template.id = "minimal"
    mock_template.render.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    mock_pm = MagicMock()
    mock_pm.get.return_value = mock_template
    mock_pm_cls.return_value = mock_pm

    runner = ExperimentRunner(
        num_test_queries=2,
        num_candidates=5,
        random_seed=42,
    )
    log = runner.run()

    # Verify that a results JSON file was written
    results_files = list(tmp_results.glob("experiment_*.json"))
    assert len(results_files) == 1, "Expected exactly one results file"

    with open(results_files[0]) as f:
        saved = json.load(f)

    assert saved["timestamp"] == log["timestamp"]


@patch("src.experiment.WikidataResolver")
@patch("src.experiment.LLMClient")
@patch("src.experiment.PromptManager")
@patch("src.experiment.EmbeddingBaseline")
@patch("src.experiment.load_fb15k237")
def test_run_config_stored_in_log(
    mock_load,
    mock_embedding_cls,
    mock_pm_cls,
    mock_llm_cls,
    mock_resolver_cls,
    tmp_results,
):
    dataset = _make_dataset()
    mock_load.return_value = dataset

    mock_embedding = MagicMock()
    mock_embedding.train.return_value = {}
    mock_embedding.get_score_fn.return_value = lambda h, r, t: 0.5
    mock_embedding_cls.return_value = mock_embedding

    mock_llm = MagicMock()
    mock_llm.chat_completion_json.return_value = {
        "parsed_json": {"score": 0.5},
        "content": '{"score": 0.5}',
        "usage": {},
        "latency_s": 0.01,
    }
    mock_llm.stats.return_value = {
        "total_calls": 3,
        "total_failures": 0,
        "total_tokens": 100,
        "total_latency_s": 0.3,
        "avg_latency_s": 0.1,
    }
    mock_llm_cls.return_value = mock_llm

    mock_resolver = MagicMock()
    mock_resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": "",
    }
    mock_resolver_cls.return_value = mock_resolver

    mock_template = MagicMock()
    mock_template.id = "minimal"
    mock_template.render.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    mock_pm = MagicMock()
    mock_pm.get.return_value = mock_template
    mock_pm_cls.return_value = mock_pm

    runner = ExperimentRunner(
        template_id="minimal",
        embedding_epochs=3,
        embedding_dim=32,
        num_test_queries=2,
        num_candidates=5,
        random_seed=7,
    )
    log = runner.run()

    cfg = log["config"]
    assert cfg["template_id"] == "minimal"
    assert cfg["embedding_epochs"] == 3
    assert cfg["embedding_dim"] == 32
    assert cfg["num_test_queries"] == 2
    assert cfg["num_candidates"] == 5
    assert cfg["random_seed"] == 7


# ---------------------------------------------------------------------------
# _print_comparison tests
# ---------------------------------------------------------------------------


def test_print_comparison_does_not_crash(capsys):
    runner = ExperimentRunner(num_test_queries=2, num_candidates=5)

    # Set up a minimal reranker mock for the LLM calls stat
    mock_scorer = MagicMock()
    mock_scorer.stats.return_value = {
        "total_scored": 0,
        "cache_hits": 0,
        "template_id": "minimal",
        "llm_stats": {"total_calls": 3},
    }
    from src.models.reranker import LLMReranker

    runner.reranker = LLMReranker(scorer=mock_scorer)
    runner.experiment_log = {"elapsed_s": 1.23}

    emb = {"MRR": 0.5, "Hits@1": 0.3, "Hits@3": 0.5, "Hits@10": 0.8}
    llm = {"MRR": 0.6, "Hits@1": 0.4, "Hits@3": 0.6, "Hits@10": 0.9}
    runner._print_comparison(emb, llm)

    captured = capsys.readouterr()
    assert "EXPERIMENT RESULTS COMPARISON" in captured.out
    assert "MRR" in captured.out
    assert "Embedding" in captured.out
    assert "LLM Reranker" in captured.out
