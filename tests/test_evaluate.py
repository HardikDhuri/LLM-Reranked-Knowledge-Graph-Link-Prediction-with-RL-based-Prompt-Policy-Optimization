"""Tests for src/eval/evaluate.py — rank_tail_query."""

from __future__ import annotations

from src.eval.evaluate import rank_tail_query
from src.eval.metrics import RankingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENTITIES = [f"e{i}" for i in range(10)]


def _score_fn_prefers(winner: str):
    """Returns a score_fn that scores `winner` highest."""

    def score_fn(h: str, r: str, t: str) -> float:
        return 1.0 if t == winner else 0.0

    return score_fn


# ---------------------------------------------------------------------------
# rank_tail_query – basic structure
# ---------------------------------------------------------------------------


def test_rank_tail_query_returns_ranking_result():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    assert isinstance(result, RankingResult)


def test_rank_tail_query_stores_query():
    query = ("h", "r", "e0")
    result = rank_tail_query(
        query=query,
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    assert result.query == query


def test_rank_tail_query_num_candidates_positive():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    assert result.num_candidates >= 1


# ---------------------------------------------------------------------------
# rank_tail_query – ranking correctness
# ---------------------------------------------------------------------------


def test_rank_tail_query_perfect_prediction_rank_1():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    assert result.true_rank == 1


def test_rank_tail_query_worst_prediction_rank_last():
    def worst_score(h: str, r: str, t: str) -> float:
        return 0.0 if t == "e0" else 1.0

    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=worst_score,
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    assert result.true_rank == result.num_candidates


def test_rank_tail_query_scored_candidates_sorted_descending():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=42,
    )
    scores = [s for _, s in result.scored_candidates]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# rank_tail_query – filtered setting
# ---------------------------------------------------------------------------


def test_rank_tail_query_filtered_removes_known_triples():
    # e1 and e2 are known, true tail is e0
    known = {("h", "r", "e1"), ("h", "r", "e2")}
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=known,
        num_candidates=8,
        seed=0,
    )
    candidate_entities = [t for t, _ in result.scored_candidates]
    assert "e1" not in candidate_entities
    assert "e2" not in candidate_entities
    assert "e0" in candidate_entities


def test_rank_tail_query_true_tail_always_in_candidates():
    result = rank_tail_query(
        query=("h", "r", "e9"),
        score_fn=_score_fn_prefers("e9"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=3,
        seed=99,
    )
    candidate_entities = [t for t, _ in result.scored_candidates]
    assert "e9" in candidate_entities


# ---------------------------------------------------------------------------
# rank_tail_query – edge cases
# ---------------------------------------------------------------------------


def test_rank_tail_query_single_entity_list():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=["e0"],
        known_triples=set(),
        num_candidates=5,
        seed=0,
    )
    assert result.true_rank == 1
    assert result.num_candidates == 1


def test_rank_tail_query_deterministic_with_seed():
    kwargs = dict(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
        seed=7,
    )
    r1 = rank_tail_query(**kwargs)
    r2 = rank_tail_query(**kwargs)
    assert r1.true_rank == r2.true_rank
    assert r1.num_candidates == r2.num_candidates


def test_rank_tail_query_no_seed_still_works():
    result = rank_tail_query(
        query=("h", "r", "e0"),
        score_fn=_score_fn_prefers("e0"),
        all_entities=ENTITIES,
        known_triples=set(),
        num_candidates=5,
    )
    assert result.true_rank >= 1
