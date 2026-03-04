"""Evaluation harness for KG link prediction queries."""

from __future__ import annotations

from typing import Callable

from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import RankingResult, compute_all_metrics, format_metrics


def rank_tail_query(
    query: tuple[str, str, str],
    score_fn: Callable[[str, str, str], float],
    all_entities: list[str],
    known_triples: set[tuple[str, str, str]],
    num_candidates: int = 50,
    seed: int | None = None,
) -> RankingResult:
    """
    For a query (h, r, true_t):
    1) Generate candidate tails
    2) Filter known triples (filtered setting)
    3) Score each candidate using score_fn(h, r, t)
    4) Rank and return result
    """
    h, r, true_t = query
    cands = generate_tail_candidates(h, r, true_t, all_entities, num_candidates, seed=seed)
    cands = filter_candidates_tail(h, r, cands, true_t, known_triples)

    scored = [(t, score_fn(h, r, t)) for t in cands]
    scored.sort(key=lambda x: x[1], reverse=True)

    ranked_entities = [t for t, _ in scored]
    true_rank = ranked_entities.index(true_t) + 1

    return RankingResult(
        query=query,
        true_rank=true_rank,
        num_candidates=len(cands),
        scored_candidates=scored,
    )


__all__ = ["rank_tail_query", "compute_all_metrics", "format_metrics"]
