"""Unit tests for src/models/scorer.py."""

from unittest.mock import MagicMock

import pytest

from src.models.scorer import TripleScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scorer(llm_response=None):
    """Return a TripleScorer with mocked LLM and WikidataResolver."""
    llm = MagicMock()
    resolver = MagicMock()
    template = MagicMock()
    template.id = "minimal"

    # Default resolver behaviour
    resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": f"description of {mid}",
    }

    # Default template behaviour
    template.render.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]

    # Default LLM behaviour
    if llm_response is None:
        llm_response = {
            "parsed_json": {"score": 0.8},
            "content": "",
            "usage": {},
            "latency_s": 0.1,
        }
    llm.chat_completion_json.return_value = llm_response

    return TripleScorer(llm_client=llm, resolver=resolver, template=template)


# ---------------------------------------------------------------------------
# score_triple
# ---------------------------------------------------------------------------


def test_score_triple_returns_float_score():
    scorer = _make_scorer(
        {"parsed_json": {"score": 0.8}, "content": "", "usage": {}, "latency_s": 0.1}
    )
    score = scorer.score_triple("/m/head", "/rel/path", "/m/tail")
    assert isinstance(score, float)
    assert score == pytest.approx(0.8)


def test_score_triple_calls_llm_once():
    scorer = _make_scorer()
    scorer.score_triple("/m/h", "/r", "/m/t")
    scorer.llm.chat_completion_json.assert_called_once()


def test_score_triple_uses_cache_on_second_call():
    scorer = _make_scorer()
    score1 = scorer.score_triple("/m/h", "/r", "/m/t")
    score2 = scorer.score_triple("/m/h", "/r", "/m/t")
    # LLM should only be called once
    scorer.llm.chat_completion_json.assert_called_once()
    assert score1 == score2
    assert scorer.total_cache_hits == 1


def test_score_triple_cache_miss_on_different_triple():
    scorer = _make_scorer()
    scorer.score_triple("/m/h", "/r", "/m/t1")
    scorer.score_triple("/m/h", "/r", "/m/t2")
    assert scorer.llm.chat_completion_json.call_count == 2


def test_score_triple_returns_zero_on_llm_failure():
    scorer = _make_scorer()
    scorer.llm.chat_completion_json.side_effect = RuntimeError("API down")
    score = scorer.score_triple("/m/h", "/r", "/m/t")
    assert score == 0.0


def test_score_triple_returns_zero_when_no_parsed_json():
    scorer = _make_scorer(
        {"parsed_json": None, "content": "oops", "usage": {}, "latency_s": 0.1}
    )
    score = scorer.score_triple("/m/h", "/r", "/m/t")
    assert score == 0.0


def test_score_triple_increments_total_scored():
    scorer = _make_scorer()
    scorer.score_triple("/m/h", "/r", "/m/t1")
    scorer.score_triple("/m/h", "/r", "/m/t2")
    assert scorer.total_scored == 2


def test_score_triple_bypass_cache():
    scorer = _make_scorer()
    scorer.score_triple("/m/h", "/r", "/m/t")
    scorer.score_triple("/m/h", "/r", "/m/t", use_cache=False)
    assert scorer.llm.chat_completion_json.call_count == 2


# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parsed,expected",
    [
        ({"score": 0.8, "reason": "looks good"}, 0.8),
        ({"score": 0}, 0.0),
        ({"score": 1.0}, 1.0),
        ({"judgment": "true", "confidence": 0.9}, 0.9),
        ({"judgment": "false", "confidence": 0.7}, 0.3),
        ({"judgment": "True", "confidence": 0.6}, 0.6),
        ({"judgment": "FALSE", "confidence": 0.4}, 0.6),
        ({}, 0.0),
        ({"unrelated": "field"}, 0.0),
    ],
)
def test_extract_score(parsed, expected):
    result = TripleScorer._extract_score(parsed)
    assert result == pytest.approx(expected, abs=1e-9)


def test_extract_score_invalid_score_type_falls_through():
    # Non-numeric score falls through to 0.0
    result = TripleScorer._extract_score({"score": "bad"})
    assert result == 0.0


# ---------------------------------------------------------------------------
# _clean_relation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relation,expected",
    [
        ("/people/person/nationality", "person nationality"),
        ("/film/film/genre", "film genre"),
        ("/music/artist/genre", "artist genre"),
        ("single_part", "single part"),
        ("/a/b/c/d", "c d"),
    ],
)
def test_clean_relation(relation, expected):
    assert TripleScorer._clean_relation(relation) == expected


# ---------------------------------------------------------------------------
# stats & clear_cache
# ---------------------------------------------------------------------------


def test_stats_returns_expected_keys():
    scorer = _make_scorer()
    scorer.llm.stats.return_value = {"total_calls": 0}
    s = scorer.stats()
    assert "total_scored" in s
    assert "cache_hits" in s
    assert "template_id" in s
    assert "llm_stats" in s
    assert s["template_id"] == "minimal"


def test_clear_cache_empties_cache():
    scorer = _make_scorer()
    scorer.score_triple("/m/h", "/r", "/m/t")
    assert len(scorer._score_cache) == 1
    scorer.clear_cache()
    assert len(scorer._score_cache) == 0
