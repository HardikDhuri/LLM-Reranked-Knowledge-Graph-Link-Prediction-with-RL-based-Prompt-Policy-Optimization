"""Unit tests for src/models/reranker.py."""

from unittest.mock import MagicMock

from src.eval.metrics import RankingResult
from src.models.reranker import LLMReranker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reranker(score_map=None):
    """Return an LLMReranker with a mocked TripleScorer.

    score_map: dict mapping tail -> score. Tails not in map get 0.0.
    """
    scorer = MagicMock()
    scorer.stats.return_value = {"total_scored": 0, "cache_hits": 0}

    if score_map is not None:
        scorer.score_triple.side_effect = lambda h, r, t: score_map.get(t, 0.0)
    else:
        scorer.score_triple.return_value = 0.5

    reranker = LLMReranker(scorer=scorer)
    return reranker


# ---------------------------------------------------------------------------
# rerank_tail_candidates
# ---------------------------------------------------------------------------


def test_rerank_returns_ranking_result():
    reranker = _make_reranker()
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2"],
    )
    assert isinstance(result, RankingResult)


def test_rerank_candidates_sorted_descending():
    score_map = {"/m/t0": 0.9, "/m/t1": 0.5, "/m/t2": 0.1}
    reranker = _make_reranker(score_map)
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2"],
    )
    scores = [s for _, s in result.scored_candidates]
    assert scores == sorted(scores, reverse=True)


def test_rerank_true_tail_highest_score_gets_rank_1():
    score_map = {"/m/t0": 0.9, "/m/t1": 0.5, "/m/t2": 0.1}
    reranker = _make_reranker(score_map)
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2"],
    )
    assert result.true_rank == 1


def test_rerank_true_tail_lowest_score_gets_last_rank():
    score_map = {"/m/t0": 0.1, "/m/t1": 0.9, "/m/t2": 0.5}
    reranker = _make_reranker(score_map)
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2"],
    )
    assert result.true_rank == 3


def test_rerank_true_tail_missing_gets_rank_beyond_candidates():
    reranker = _make_reranker()
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/missing",
        candidates=["/m/t1", "/m/t2"],
    )
    assert result.true_rank == 3  # len(candidates) + 1


def test_rerank_correct_num_candidates():
    reranker = _make_reranker()
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2", "/m/t3"],
    )
    assert result.num_candidates == 4


def test_rerank_query_stored_in_result():
    reranker = _make_reranker()
    result = reranker.rerank_tail_candidates(
        head="/m/h",
        relation="/r",
        true_tail="/m/t0",
        candidates=["/m/t0"],
    )
    assert result.query == ("/m/h", "/r", "/m/t0")


def test_rerank_increments_total_queries():
    reranker = _make_reranker()
    reranker.rerank_tail_candidates("/m/h", "/r", "/m/t", ["/m/t"])
    reranker.rerank_tail_candidates("/m/h", "/r", "/m/t", ["/m/t"])
    assert reranker.total_queries == 2


# ---------------------------------------------------------------------------
# rerank_batch
# ---------------------------------------------------------------------------


def test_rerank_batch_processes_multiple_queries():
    score_map = {"/m/t0": 0.9, "/m/t1": 0.2}
    reranker = _make_reranker(score_map)
    queries = [
        ("/m/h1", "/r", "/m/t0"),
        ("/m/h2", "/r", "/m/t1"),
    ]
    candidates_per_query = {
        ("/m/h1", "/r", "/m/t0"): ["/m/t0", "/m/t1"],
        ("/m/h2", "/r", "/m/t1"): ["/m/t0", "/m/t1"],
    }
    results = reranker.rerank_batch(queries, candidates_per_query, progress=False)
    assert len(results) == 2


def test_rerank_batch_skips_queries_with_no_candidates():
    reranker = _make_reranker()
    queries = [
        ("/m/h1", "/r", "/m/t0"),
        ("/m/h2", "/r", "/m/t1"),
    ]
    # Only provide candidates for the first query
    candidates_per_query = {
        ("/m/h1", "/r", "/m/t0"): ["/m/t0"],
    }
    results = reranker.rerank_batch(queries, candidates_per_query, progress=False)
    assert len(results) == 1
    assert results[0].query == ("/m/h1", "/r", "/m/t0")


def test_rerank_batch_returns_empty_when_no_candidates():
    reranker = _make_reranker()
    queries = [("/m/h", "/r", "/m/t")]
    results = reranker.rerank_batch(queries, {}, progress=False)
    assert results == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_correct_keys():
    reranker = _make_reranker()
    s = reranker.stats()
    assert "total_queries" in s
    assert "total_rerank_time_s" in s
    assert "avg_time_per_query_s" in s
    assert "scorer_stats" in s


def test_stats_total_queries_count():
    reranker = _make_reranker()
    reranker.rerank_tail_candidates("/m/h", "/r", "/m/t", ["/m/t"])
    reranker.rerank_tail_candidates("/m/h", "/r", "/m/t", ["/m/t"])
    assert reranker.stats()["total_queries"] == 2


def test_stats_avg_time_safe_when_no_queries():
    reranker = _make_reranker()
    s = reranker.stats()
    assert s["avg_time_per_query_s"] == 0.0
