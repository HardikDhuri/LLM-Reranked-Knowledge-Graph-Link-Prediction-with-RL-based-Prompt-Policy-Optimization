"""Tests for src/eval/metrics.py."""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    RankingResult,
    compute_all_metrics,
    format_metrics,
    hits_at_k,
    mean_reciprocal_rank,
)


def _make_result(rank: int, num_candidates: int = 10) -> RankingResult:
    return RankingResult(
        query=("h", "r", "t"),
        true_rank=rank,
        num_candidates=num_candidates,
    )


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


def test_mrr_all_rank_one() -> None:
    results = [_make_result(1), _make_result(1), _make_result(1)]
    assert mean_reciprocal_rank(results) == pytest.approx(1.0)


def test_mrr_ranks_1_2_3() -> None:
    results = [_make_result(1), _make_result(2), _make_result(3)]
    expected = (1.0 + 0.5 + 1.0 / 3) / 3
    assert mean_reciprocal_rank(results) == pytest.approx(expected)


def test_mrr_empty() -> None:
    assert mean_reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# hits_at_k
# ---------------------------------------------------------------------------


def test_hits_at_k_partial() -> None:
    results = [_make_result(1), _make_result(2), _make_result(5)]
    assert hits_at_k(results, k=3) == pytest.approx(2 / 3)


def test_hits_at_k_all_rank_one() -> None:
    results = [_make_result(1), _make_result(1), _make_result(1)]
    assert hits_at_k(results, k=1) == pytest.approx(1.0)


def test_hits_at_k_empty() -> None:
    assert hits_at_k([], k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_compute_all_metrics_keys() -> None:
    results = [_make_result(1), _make_result(2)]
    metrics = compute_all_metrics(results, ks=(1, 3, 10))
    assert "MRR" in metrics
    assert "Hits@1" in metrics
    assert "Hits@3" in metrics
    assert "Hits@10" in metrics
    assert "num_queries" in metrics
    assert "avg_candidates" in metrics


def test_compute_all_metrics_values() -> None:
    results = [_make_result(1, num_candidates=10), _make_result(2, num_candidates=20)]
    metrics = compute_all_metrics(results, ks=(1,))
    assert metrics["MRR"] == pytest.approx((1.0 + 0.5) / 2)
    assert metrics["Hits@1"] == pytest.approx(0.5)
    assert metrics["num_queries"] == 2
    assert metrics["avg_candidates"] == pytest.approx(15.0)


def test_compute_all_metrics_empty() -> None:
    metrics = compute_all_metrics([], ks=(1, 3, 10))
    assert metrics["MRR"] == 0.0
    assert metrics["avg_candidates"] == 0


# ---------------------------------------------------------------------------
# format_metrics
# ---------------------------------------------------------------------------


def test_format_metrics_returns_string() -> None:
    results = [_make_result(1)]
    metrics = compute_all_metrics(results)
    output = format_metrics(metrics)
    assert isinstance(output, str)
    assert "MRR" in output
