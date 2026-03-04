"""Unit tests for src/rl/budget_experiment.py — BudgetExperiment.

All external dependencies (dataset, embedding training, LLM, Wikidata)
are fully mocked so tests run without internet access or real data.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data.fb15k237 import FB15k237Dataset
from src.eval.metrics import RankingResult
from src.rl.budget_experiment import BudgetExperiment

# ---------------------------------------------------------------------------
# Tiny synthetic dataset
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


def _make_dataset() -> FB15k237Dataset:
    return FB15k237Dataset(train=_TRAIN, valid=_VALID, test=_TEST)


def _make_ranking_result(query: tuple) -> RankingResult:
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
def tmp_results(tmp_path, monkeypatch):
    """Redirect results_dir to a tmp_path and clear settings cache."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    from src.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helper to build a fully mocked experiment run
# ---------------------------------------------------------------------------


def _run_mocked_experiment(tmp_results, num_queries: int = 2, budget: int = 5):
    """Patch all external dependencies and run BudgetExperiment."""
    dataset = _make_dataset()

    with (
        patch("src.rl.budget_experiment.load_fb15k237", return_value=dataset),
        patch("src.rl.budget_experiment.WikidataResolver") as mock_resolver_cls,
        patch("src.rl.budget_experiment.LLMClient") as mock_llm_cls,
        patch("src.rl.budget_experiment.PromptManager") as mock_pm_cls,
        patch("src.rl.budget_experiment.EmbeddingBaseline") as mock_emb_cls,
        patch("src.rl.budget_experiment.LLMReranker") as mock_reranker_cls,
        patch("src.rl.budget_experiment.TripleScorer"),
    ):
        # Mock embedding
        mock_emb = MagicMock()
        mock_emb.train.return_value = {"model": "RotatE"}
        mock_emb.get_score_fn.return_value = lambda h, r, t: 0.5
        mock_emb_cls.return_value = mock_emb

        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # Mock WikidataResolver
        mock_resolver = MagicMock()
        mock_resolver_cls.return_value = mock_resolver

        # Mock PromptManager
        mock_template = MagicMock()
        mock_template.id = "minimal"
        mock_pm = MagicMock()
        mock_pm.get.return_value = mock_template
        mock_pm_cls.return_value = mock_pm

        # Mock LLMReranker to return a RankingResult
        mock_reranker = MagicMock()
        mock_reranker.rerank_tail_candidates.return_value = RankingResult(
            query=("e0", "r0", "e2"),
            true_rank=1,
            num_candidates=5,
            scored_candidates=[("e2", 0.9), ("e1", 0.1)],
        )
        mock_reranker_cls.return_value = mock_reranker

        experiment = BudgetExperiment(
            template_id="minimal",
            total_budget=budget,
            agent_type="linucb",
            num_test_queries=num_queries,
            num_candidates=5,
            random_seed=42,
        )
        result = experiment.run()

    return result


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_default_values(tmp_results):
    from src.config import get_settings

    settings = get_settings()
    experiment = BudgetExperiment()
    assert experiment.template_id == "minimal"
    assert experiment.total_budget == 50
    assert experiment.agent_type == "linucb"
    assert experiment.num_test_queries == settings.sample_test_queries
    assert experiment.num_candidates == settings.num_candidates


def test_init_custom_values(tmp_results):
    experiment = BudgetExperiment(
        template_id="chain_of_thought",
        total_budget=20,
        agent_type="epsilon_greedy",
        num_test_queries=5,
        num_candidates=10,
        random_seed=99,
    )
    assert experiment.template_id == "chain_of_thought"
    assert experiment.total_budget == 20
    assert experiment.agent_type == "epsilon_greedy"
    assert experiment.num_test_queries == 5
    assert experiment.num_candidates == 10
    assert experiment.random_seed == 99


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


def test_run_returns_dict_with_expected_keys(tmp_results):
    result = _run_mocked_experiment(tmp_results, num_queries=2, budget=5)
    assert isinstance(result, dict)
    for key in [
        "timestamp",
        "config",
        "embedding_metrics",
        "rl_budget_metrics",
        "budget_agent_summary",
        "elapsed_s",
    ]:
        assert key in result, f"Missing key: {key}"


def test_run_saves_results_file(tmp_results):
    _run_mocked_experiment(tmp_results, num_queries=2, budget=5)
    result_files = list(tmp_results.glob("budget_experiment_*.json"))
    assert len(result_files) == 1, "Expected exactly one results file"

    with open(result_files[0]) as f:
        saved = json.load(f)

    assert "timestamp" in saved
    assert "config" in saved
    assert "embedding_metrics" in saved
    assert "rl_budget_metrics" in saved
    assert "budget_agent_summary" in saved


def test_run_config_stored_correctly(tmp_results):
    result = _run_mocked_experiment(tmp_results, num_queries=2, budget=5)
    cfg = result["config"]
    assert cfg["template_id"] == "minimal"
    assert cfg["total_budget"] == 5
    assert cfg["agent_type"] == "linucb"
    assert cfg["num_test_queries"] == 2
    assert cfg["num_candidates"] == 5


def test_run_both_metrics_computed(tmp_results):
    result = _run_mocked_experiment(tmp_results, num_queries=2, budget=5)
    emb = result["embedding_metrics"]
    rl = result["rl_budget_metrics"]
    assert "MRR" in emb
    assert "Hits@1" in emb
    assert "MRR" in rl
    assert "Hits@1" in rl


def test_run_budget_agent_summary_correct(tmp_results):
    result = _run_mocked_experiment(tmp_results, num_queries=2, budget=5)
    summary = result["budget_agent_summary"]
    assert "total_budget" in summary
    assert "remaining_budget" in summary
    assert "llm_queries" in summary
    assert "embedding_queries" in summary
    assert summary["total_budget"] == 5
    assert summary["total_queries"] == 2
    assert summary["llm_queries"] + summary["embedding_queries"] == 2


def test_run_budget_exhaustion_falls_back_to_embedding(tmp_results):
    """When budget=0, all queries should use embedding-only."""
    result = _run_mocked_experiment(tmp_results, num_queries=3, budget=0)
    summary = result["budget_agent_summary"]
    assert summary["llm_queries"] == 0
    assert summary["embedding_queries"] == 3


# ---------------------------------------------------------------------------
# _print_comparison tests
# ---------------------------------------------------------------------------


def test_print_comparison_does_not_crash(capsys, tmp_results):
    from src.rl.budget_agent import BudgetAgent
    from src.rl.features import QueryFeatureExtractor

    extractor = QueryFeatureExtractor.from_triples(_TRAIN)
    agent = BudgetAgent(feature_extractor=extractor, total_budget=10)

    experiment = BudgetExperiment(num_test_queries=2, num_candidates=5)
    emb = {"MRR": 0.5, "Hits@1": 0.3, "Hits@3": 0.5, "Hits@10": 0.8}
    rl = {"MRR": 0.6, "Hits@1": 0.4, "Hits@3": 0.6, "Hits@10": 0.9}
    experiment._print_comparison(emb, rl, agent)

    captured = capsys.readouterr()
    assert "BUDGET EXPERIMENT RESULTS" in captured.out
    assert "MRR" in captured.out
    assert "Embedding-Only" in captured.out
    assert "RL Budget" in captured.out
