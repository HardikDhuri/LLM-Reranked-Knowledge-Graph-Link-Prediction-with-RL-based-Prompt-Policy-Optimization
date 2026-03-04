"""Ranking evaluation metrics: MRR and Hits@K."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankingResult:
    """Result of ranking candidates for a single query."""

    query: tuple[str, str, str]  # (h, r, true_t) or (true_h, r, t)
    true_rank: int  # 1-based rank of the true entity
    num_candidates: int  # total candidates ranked
    scored_candidates: list[tuple[str, float]] = field(
        default_factory=list
    )  # (entity, score) sorted desc


def mean_reciprocal_rank(results: list[RankingResult]) -> float:
    """Compute MRR over a list of ranking results."""
    if not results:
        return 0.0
    return sum(1.0 / r.true_rank for r in results) / len(results)


def hits_at_k(results: list[RankingResult], k: int) -> float:
    """Compute Hits@K (fraction of queries where true entity is ranked <= k)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.true_rank <= k) / len(results)


def compute_all_metrics(
    results: list[RankingResult], ks: tuple[int, ...] = (1, 3, 10)
) -> dict[str, float]:
    """Compute MRR + Hits@K for multiple K values."""
    metrics: dict[str, float] = {"MRR": mean_reciprocal_rank(results)}
    for k in ks:
        metrics[f"Hits@{k}"] = hits_at_k(results, k)
    metrics["num_queries"] = len(results)
    metrics["avg_candidates"] = (
        sum(r.num_candidates for r in results) / len(results) if results else 0
    )
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty-print metrics."""
    lines = []
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)
