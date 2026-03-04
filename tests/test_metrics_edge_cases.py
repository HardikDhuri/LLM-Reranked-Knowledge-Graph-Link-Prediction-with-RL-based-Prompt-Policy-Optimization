"""Edge case tests for src/eval/metrics.py."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    RankingResult,
    compute_all_metrics,
    format_metrics,
    hits_at_k,
    mean_reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(rank: int, num_candidates: int = 10) -> RankingResult:
    return RankingResult(
        query=("h", "r", "t"),
        true_rank=rank,
        num_candidates=num_candidates,
    )


# ---------------------------------------------------------------------------
# compute_all_metrics – edge cases
# ---------------------------------------------------------------------------


def test_compute_all_metrics_single_result():
    results = [_result(1)]
    metrics = compute_all_metrics(results)
    assert metrics["MRR"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 1


def test_compute_all_metrics_multiple_results():
    results = [_result(1), _result(2), _result(5)]
    metrics = compute_all_metrics(results, ks=(1, 3, 10))
    assert metrics["num_queries"] == 3
    assert metrics["MRR"] == pytest.approx((1.0 + 0.5 + 0.2) / 3)
    assert metrics["Hits@1"] == pytest.approx(1 / 3)
    assert metrics["Hits@3"] == pytest.approx(2 / 3)
    assert metrics["Hits@10"] == pytest.approx(1.0)


def test_compute_all_metrics_empty_returns_zeros():
    metrics = compute_all_metrics([])
    assert metrics["MRR"] == 0.0
    assert metrics["num_queries"] == 0
    assert metrics["avg_candidates"] == 0


def test_compute_all_metrics_all_rank_1_perfect():
    results = [_result(1), _result(1), _result(1)]
    metrics = compute_all_metrics(results, ks=(1, 3, 10))
    assert metrics["MRR"] == pytest.approx(1.0)
    assert metrics["Hits@1"] == pytest.approx(1.0)
    assert metrics["Hits@3"] == pytest.approx(1.0)
    assert metrics["Hits@10"] == pytest.approx(1.0)


def test_compute_all_metrics_all_rank_last_worst():
    n = 20
    results = [_result(n, num_candidates=n) for _ in range(5)]
    metrics = compute_all_metrics(results, ks=(1, 3, 10))
    assert metrics["MRR"] == pytest.approx(1.0 / n)
    assert metrics["Hits@1"] == pytest.approx(0.0)
    assert metrics["Hits@3"] == pytest.approx(0.0)
    assert metrics["Hits@10"] == pytest.approx(0.0)


def test_compute_all_metrics_avg_candidates():
    results = [_result(1, num_candidates=10), _result(2, num_candidates=20)]
    metrics = compute_all_metrics(results)
    assert metrics["avg_candidates"] == pytest.approx(15.0)


def test_compute_all_metrics_custom_ks():
    results = [_result(5)]
    metrics = compute_all_metrics(results, ks=(3, 5, 7))
    assert "Hits@3" in metrics
    assert "Hits@5" in metrics
    assert "Hits@7" in metrics
    assert metrics["Hits@3"] == pytest.approx(0.0)
    assert metrics["Hits@5"] == pytest.approx(1.0)
    assert metrics["Hits@7"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_metrics – output checks
# ---------------------------------------------------------------------------


def test_format_metrics_returns_string():
    metrics = compute_all_metrics([_result(1)])
    output = format_metrics(metrics)
    assert isinstance(output, str)


def test_format_metrics_contains_metric_names():
    metrics = compute_all_metrics([_result(1)], ks=(1, 3))
    output = format_metrics(metrics)
    assert "MRR" in output
    assert "Hits@1" in output
    assert "Hits@3" in output


def test_format_metrics_multiline():
    metrics = compute_all_metrics([_result(1)], ks=(1,))
    output = format_metrics(metrics)
    assert "\n" in output


def test_format_metrics_integer_values_not_formatted_as_float():
    metrics = {"num_queries": 5, "MRR": 0.75}
    output = format_metrics(metrics)
    assert "5" in output
    assert "0.7500" in output


# ---------------------------------------------------------------------------
# mean_reciprocal_rank and hits_at_k – additional edge cases
# ---------------------------------------------------------------------------


def test_mrr_single_result():
    assert mean_reciprocal_rank([_result(4)]) == pytest.approx(0.25)


def test_hits_at_k_boundary():
    results = [_result(3)]
    assert hits_at_k(results, k=3) == pytest.approx(1.0)
    assert hits_at_k(results, k=2) == pytest.approx(0.0)


def test_hits_at_k_all_tie():
    results = [_result(2), _result(2), _result(2)]
    assert hits_at_k(results, k=1) == pytest.approx(0.0)
    assert hits_at_k(results, k=2) == pytest.approx(1.0)
