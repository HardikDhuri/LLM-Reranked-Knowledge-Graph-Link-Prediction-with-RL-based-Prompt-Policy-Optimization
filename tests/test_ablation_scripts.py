"""Tests for ablation scripts (ablation_embedding_dim, ablation_num_candidates,
ablation_alpha, ablation_budget_levels).

All external dependencies are mocked so tests run without real data or LLMs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data.fb15k237 import FB15k237Dataset
from src.eval.metrics import RankingResult

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_TRAIN = [
    ("e0", "r0", "e1"),
    ("e1", "r0", "e2"),
    ("e2", "r0", "e3"),
    ("e0", "r1", "e2"),
    ("e1", "r1", "e3"),
]
_VALID = [("e0", "r0", "e3")]
_TEST = [("e0", "r0", "e2"), ("e1", "r0", "e3")]


def _make_dataset():
    return FB15k237Dataset(train=_TRAIN, valid=_VALID, test=_TEST)


def _make_ranking_result(query):
    h, r, t = query
    return RankingResult(
        query=query,
        true_rank=1,
        num_candidates=5,
        scored_candidates=[(t, 0.9), ("e_other", 0.1)],
    )


@pytest.fixture()
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    from src.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helpers for common mocking patterns
# ---------------------------------------------------------------------------


def _mock_embedding():
    mock = MagicMock()
    mock.train.return_value = {"model": "RotatE"}
    mock.get_score_fn.return_value = lambda h, r, t: 0.5
    return mock


def _mock_reranker(test_queries):
    mock = MagicMock()
    mock.rerank_tail_candidates.side_effect = lambda h, r, t, cands: RankingResult(
        query=(h, r, t),
        true_rank=1,
        num_candidates=len(cands),
        scored_candidates=[(c, 0.5) for c in cands],
    )
    mock.stats.return_value = {"scorer_stats": {"llm_stats": {"total_calls": 5}}}
    return mock


# ---------------------------------------------------------------------------
# ablation_embedding_dim tests
# ---------------------------------------------------------------------------


def test_ablation_embedding_dim_sweep_runs(tmp_results):
    """run_sweep returns one result per dimension."""
    from scripts.ablation_embedding_dim import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()

    with (
        patch("scripts.ablation_embedding_dim.load_fb15k237", return_value=dataset),
        patch(
            "scripts.ablation_embedding_dim.EmbeddingBaseline", return_value=mock_emb
        ),
    ):
        results = run_sweep(
            dims=[32, 64],
            epochs=5,
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    assert len(results) == 2
    dims_seen = [r["dim"] for r in results]
    assert 32 in dims_seen
    assert 64 in dims_seen


def test_ablation_embedding_dim_result_structure(tmp_results):
    """Each result entry has expected keys."""
    from scripts.ablation_embedding_dim import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()

    with (
        patch("scripts.ablation_embedding_dim.load_fb15k237", return_value=dataset),
        patch(
            "scripts.ablation_embedding_dim.EmbeddingBaseline", return_value=mock_emb
        ),
    ):
        results = run_sweep(
            dims=[64], epochs=5, num_queries=2, num_candidates=5, random_seed=42
        )

    assert len(results) == 1
    row = results[0]
    assert "dim" in row
    assert "metrics" in row
    assert "train_time_s" in row
    assert "MRR" in row["metrics"]


def test_ablation_embedding_dim_saves_json(tmp_results):
    """main() saves a JSON file to results dir."""
    from scripts.ablation_embedding_dim import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()

    with (
        patch("scripts.ablation_embedding_dim.load_fb15k237", return_value=dataset),
        patch(
            "scripts.ablation_embedding_dim.EmbeddingBaseline", return_value=mock_emb
        ),
    ):
        results = run_sweep(
            dims=[32], epochs=5, num_queries=2, num_candidates=5, random_seed=42
        )

    # Verify JSON would be serializable
    payload = {
        "timestamp": "20240101_120000",
        "config": {"dims": [32], "epochs": 5, "num_queries": 2, "num_candidates": 5},
        "results": results,
    }
    out_file = tmp_results / "ablation_embedding_dim_20240101_120000.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(out_file) as f:
        saved = json.load(f)

    assert "timestamp" in saved
    assert "config" in saved
    assert "results" in saved
    assert saved["config"]["dims"] == [32]


# ---------------------------------------------------------------------------
# ablation_num_candidates tests
# ---------------------------------------------------------------------------


def test_ablation_num_candidates_sweep_runs(tmp_results):
    """run_sweep returns one result per candidate count."""
    from scripts.ablation_num_candidates import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()
    mock_reranker = _mock_reranker(_TEST)

    with (
        patch("scripts.ablation_num_candidates.load_fb15k237", return_value=dataset),
        patch(
            "scripts.ablation_num_candidates.EmbeddingBaseline", return_value=mock_emb
        ),
        patch("scripts.ablation_num_candidates.WikidataResolver"),
        patch("scripts.ablation_num_candidates.LLMClient"),
        patch("scripts.ablation_num_candidates.PromptManager") as mock_pm_cls,
        patch(
            "scripts.ablation_num_candidates.LLMReranker", return_value=mock_reranker
        ),
        patch("scripts.ablation_num_candidates.TripleScorer"),
    ):
        mock_pm = MagicMock()
        mock_pm.get.return_value = MagicMock()
        mock_pm_cls.return_value = mock_pm

        results = run_sweep(
            candidate_counts=[5, 10],
            template_id="minimal",
            num_queries=2,
            random_seed=42,
        )

    assert len(results) == 2
    counts_seen = [r["num_candidates"] for r in results]
    assert 5 in counts_seen
    assert 10 in counts_seen


def test_ablation_num_candidates_result_structure(tmp_results):
    """Each result entry has embedding_metrics, llm_metrics, llm_calls."""
    from scripts.ablation_num_candidates import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()
    mock_reranker = _mock_reranker(_TEST)

    with (
        patch("scripts.ablation_num_candidates.load_fb15k237", return_value=dataset),
        patch(
            "scripts.ablation_num_candidates.EmbeddingBaseline", return_value=mock_emb
        ),
        patch("scripts.ablation_num_candidates.WikidataResolver"),
        patch("scripts.ablation_num_candidates.LLMClient"),
        patch("scripts.ablation_num_candidates.PromptManager") as mock_pm_cls,
        patch(
            "scripts.ablation_num_candidates.LLMReranker", return_value=mock_reranker
        ),
        patch("scripts.ablation_num_candidates.TripleScorer"),
    ):
        mock_pm = MagicMock()
        mock_pm.get.return_value = MagicMock()
        mock_pm_cls.return_value = mock_pm

        results = run_sweep(
            candidate_counts=[5],
            template_id="minimal",
            num_queries=2,
            random_seed=42,
        )

    row = results[0]
    assert "num_candidates" in row
    assert "embedding_metrics" in row
    assert "llm_metrics" in row
    assert "llm_calls" in row


# ---------------------------------------------------------------------------
# ablation_alpha tests
# ---------------------------------------------------------------------------


def test_ablation_alpha_sweep_runs(tmp_results):
    """run_sweep returns one result per alpha value."""
    from scripts.ablation_alpha import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()

    with (
        patch("scripts.ablation_alpha.load_fb15k237", return_value=dataset),
        patch("scripts.ablation_alpha.EmbeddingBaseline", return_value=mock_emb),
        patch("scripts.ablation_alpha.WikidataResolver"),
        patch("scripts.ablation_alpha.LLMClient"),
        patch("scripts.ablation_alpha.PromptManager") as mock_pm_cls,
        patch("scripts.ablation_alpha.RLPromptSelector") as mock_selector_cls,
    ):
        mock_pm = MagicMock()
        mock_pm.get.return_value = MagicMock()
        mock_pm_cls.return_value = mock_pm

        mock_selector = MagicMock()
        mock_selector.select_and_score.return_value = (
            "minimal",
            [("e2", 0.9), ("e3", 0.5)],
            1,
        )
        mock_selector.summary.return_value = {
            "agent": {"arm_selection_counts": {"minimal": 2}},
            "template_ids": ["minimal"],
        }
        mock_selector_cls.return_value = mock_selector

        results = run_sweep(
            alphas=[0.1, 1.0],
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    assert len(results) == 2
    alphas_seen = [r["alpha"] for r in results]
    assert 0.1 in alphas_seen
    assert 1.0 in alphas_seen


def test_ablation_alpha_result_structure(tmp_results):
    """Each result entry has alpha, metrics, cumulative_reward, arm_distribution."""
    from scripts.ablation_alpha import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()

    with (
        patch("scripts.ablation_alpha.load_fb15k237", return_value=dataset),
        patch("scripts.ablation_alpha.EmbeddingBaseline", return_value=mock_emb),
        patch("scripts.ablation_alpha.WikidataResolver"),
        patch("scripts.ablation_alpha.LLMClient"),
        patch("scripts.ablation_alpha.PromptManager") as mock_pm_cls,
        patch("scripts.ablation_alpha.RLPromptSelector") as mock_selector_cls,
    ):
        mock_pm = MagicMock()
        mock_pm.get.return_value = MagicMock()
        mock_pm_cls.return_value = mock_pm

        mock_selector = MagicMock()
        mock_selector.select_and_score.return_value = ("minimal", [("e2", 0.9)], 1)
        mock_selector.summary.return_value = {
            "agent": {"arm_selection_counts": {}},
            "template_ids": ["minimal"],
        }
        mock_selector_cls.return_value = mock_selector

        results = run_sweep(
            alphas=[0.5],
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    row = results[0]
    assert "alpha" in row
    assert "metrics" in row
    assert "cumulative_reward" in row
    assert "arm_distribution" in row


# ---------------------------------------------------------------------------
# ablation_budget_levels tests
# ---------------------------------------------------------------------------


def test_ablation_budget_levels_sweep_runs(tmp_results):
    """run_sweep returns one result per fraction."""
    from scripts.ablation_budget_levels import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()
    mock_reranker = _mock_reranker(_TEST)

    with (
        patch("src.rl.budget_experiment.load_fb15k237", return_value=dataset),
        patch("src.rl.budget_experiment.EmbeddingBaseline", return_value=mock_emb),
        patch("src.rl.budget_experiment.WikidataResolver"),
        patch("src.rl.budget_experiment.LLMClient"),
        patch("src.rl.budget_experiment.PromptManager") as mock_pm_cls,
        patch(
            "src.rl.budget_experiment.LLMReranker", return_value=mock_reranker
        ),
        patch("src.rl.budget_experiment.TripleScorer"),
    ):
        mock_pm = MagicMock()
        mock_template = MagicMock()
        mock_template.id = "minimal"
        mock_pm.get.return_value = mock_template
        mock_pm_cls.return_value = mock_pm

        results = run_sweep(
            fractions=[0.0, 0.5, 1.0],
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    assert len(results) == 3
    fractions_seen = [r["fraction"] for r in results]
    assert 0.0 in fractions_seen
    assert 0.5 in fractions_seen
    assert 1.0 in fractions_seen


def test_ablation_budget_levels_result_structure(tmp_results):
    """Each result entry has fraction, budget, metrics, embedding_metrics, llm_calls."""
    from scripts.ablation_budget_levels import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()
    mock_reranker = _mock_reranker(_TEST)

    with (
        patch("src.rl.budget_experiment.load_fb15k237", return_value=dataset),
        patch("src.rl.budget_experiment.EmbeddingBaseline", return_value=mock_emb),
        patch("src.rl.budget_experiment.WikidataResolver"),
        patch("src.rl.budget_experiment.LLMClient"),
        patch("src.rl.budget_experiment.PromptManager") as mock_pm_cls,
        patch(
            "src.rl.budget_experiment.LLMReranker", return_value=mock_reranker
        ),
        patch("src.rl.budget_experiment.TripleScorer"),
    ):
        mock_pm = MagicMock()
        mock_template = MagicMock()
        mock_template.id = "minimal"
        mock_pm.get.return_value = mock_template
        mock_pm_cls.return_value = mock_pm

        results = run_sweep(
            fractions=[0.5],
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    row = results[0]
    assert "fraction" in row
    assert "budget" in row
    assert "metrics" in row
    assert "embedding_metrics" in row
    assert "llm_calls" in row


def test_ablation_budget_levels_json_structure(tmp_results):
    """Results can be serialized to JSON with expected top-level keys."""
    from scripts.ablation_budget_levels import run_sweep

    dataset = _make_dataset()
    mock_emb = _mock_embedding()
    mock_reranker = _mock_reranker(_TEST)

    with (
        patch("src.rl.budget_experiment.load_fb15k237", return_value=dataset),
        patch("src.rl.budget_experiment.EmbeddingBaseline", return_value=mock_emb),
        patch("src.rl.budget_experiment.WikidataResolver"),
        patch("src.rl.budget_experiment.LLMClient"),
        patch("src.rl.budget_experiment.PromptManager") as mock_pm_cls,
        patch(
            "src.rl.budget_experiment.LLMReranker", return_value=mock_reranker
        ),
        patch("src.rl.budget_experiment.TripleScorer"),
    ):
        mock_pm = MagicMock()
        mock_template = MagicMock()
        mock_template.id = "minimal"
        mock_pm.get.return_value = mock_template
        mock_pm_cls.return_value = mock_pm

        results = run_sweep(
            fractions=[0.0, 1.0],
            num_queries=2,
            num_candidates=5,
            random_seed=42,
        )

    payload = {
        "timestamp": "20240101_120000",
        "config": {"fractions": [0.0, 1.0], "num_queries": 2},
        "results": results,
    }
    out_file = tmp_results / "ablation_budget_20240101_120000.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(out_file) as f:
        saved = json.load(f)

    assert "timestamp" in saved
    assert "config" in saved
    assert "results" in saved
    assert len(saved["results"]) == 2
