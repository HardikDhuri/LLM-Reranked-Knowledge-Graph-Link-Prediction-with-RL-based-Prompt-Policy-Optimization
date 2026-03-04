"""Tests for src/rl/prompt_selector.py — RLPromptSelector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.rl.features import QueryFeatureExtractor
from src.rl.prompt_selector import RLPromptSelector

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TRIPLES = [
    ("/m/h1", "/rel/a", "/m/t1"),
    ("/m/h1", "/rel/a", "/m/t2"),
    ("/m/h2", "/rel/b", "/m/t1"),
]


def _make_mock_template(tid: str):
    tmpl = MagicMock()
    tmpl.id = tid
    tmpl.render.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    return tmpl


def _make_selector(agent_type: str = "linucb") -> RLPromptSelector:
    """Create an RLPromptSelector with fully mocked external dependencies."""
    llm_client = MagicMock()
    llm_client.chat_completion_json.return_value = {
        "parsed_json": {"score": 0.7},
        "content": "",
        "usage": {},
        "latency_s": 0.01,
    }

    resolver = MagicMock()
    resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": f"description of {mid}",
    }

    # Build a mock PromptManager with two templates
    template_ids = ["minimal", "verbose"]
    templates = {tid: _make_mock_template(tid) for tid in template_ids}

    prompt_manager = MagicMock()
    prompt_manager.list_ids.return_value = template_ids
    prompt_manager.get.side_effect = lambda tid: templates[tid]

    feature_extractor = QueryFeatureExtractor.from_triples(TRIPLES)

    return RLPromptSelector(
        llm_client=llm_client,
        resolver=resolver,
        prompt_manager=prompt_manager,
        feature_extractor=feature_extractor,
        agent_type=agent_type,
    )


# ---------------------------------------------------------------------------
# RLPromptSelector — construction
# ---------------------------------------------------------------------------


def test_selector_linucb_initialises():
    selector = _make_selector("linucb")
    assert selector.template_ids == ["minimal", "verbose"]
    assert len(selector.scorers) == 2


def test_selector_epsilon_greedy_initialises():
    selector = _make_selector("epsilon_greedy")
    assert selector.template_ids == ["minimal", "verbose"]
    assert len(selector.scorers) == 2


def test_selector_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent type"):
        llm_client = MagicMock()
        resolver = MagicMock()
        prompt_manager = MagicMock()
        prompt_manager.list_ids.return_value = ["minimal"]
        prompt_manager.get.return_value = _make_mock_template("minimal")
        feature_extractor = QueryFeatureExtractor.from_triples(TRIPLES)
        RLPromptSelector(
            llm_client=llm_client,
            resolver=resolver,
            prompt_manager=prompt_manager,
            feature_extractor=feature_extractor,
            agent_type="bad_agent",
        )


# ---------------------------------------------------------------------------
# select_template
# ---------------------------------------------------------------------------


def test_select_template_returns_valid_id_linucb():
    selector = _make_selector("linucb")
    tid = selector.select_template("/m/h1", "/rel/a", "/m/t1")
    assert tid in selector.template_ids


def test_select_template_returns_valid_id_epsilon_greedy():
    selector = _make_selector("epsilon_greedy")
    tid = selector.select_template("/m/h1", "/rel/a", "/m/t1")
    assert tid in selector.template_ids


# ---------------------------------------------------------------------------
# select_and_score
# ---------------------------------------------------------------------------


def test_select_and_score_returns_three_tuple():
    selector = _make_selector("linucb")
    candidates = ["/m/t1", "/m/t2", "/m/t3"]
    result = selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_select_and_score_template_id_is_valid():
    selector = _make_selector("linucb")
    candidates = ["/m/t1", "/m/t2"]
    template_id, _, _ = selector.select_and_score(
        "/m/h1", "/rel/a", "/m/t1", candidates
    )
    assert template_id in selector.template_ids


def test_select_and_score_scored_candidates_length():
    selector = _make_selector("linucb")
    candidates = ["/m/t1", "/m/t2", "/m/t3"]
    _, scored, _ = selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    assert len(scored) == len(candidates)


def test_select_and_score_true_rank_positive_int():
    selector = _make_selector("linucb")
    candidates = ["/m/t1", "/m/t2", "/m/t3"]
    _, _, true_rank = selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    assert isinstance(true_rank, int)
    assert true_rank >= 1


def test_select_and_score_true_rank_when_true_tail_not_in_candidates():
    selector = _make_selector("linucb")
    candidates = ["/m/t2", "/m/t3"]  # true tail absent
    _, _, true_rank = selector.select_and_score(
        "/m/h1", "/rel/a", "/m/t1_missing", candidates
    )
    # rank should be len(candidates) + 1 when tail is absent
    assert true_rank == len(candidates) + 1


def test_select_and_score_updates_agent_linucb():
    selector = _make_selector("linucb")
    assert selector.agent.total_steps == 0
    candidates = ["/m/t1", "/m/t2"]
    selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    assert selector.agent.total_steps == 1


def test_select_and_score_updates_agent_epsilon_greedy():
    selector = _make_selector("epsilon_greedy")
    assert selector.agent.total_steps == 0
    candidates = ["/m/t1", "/m/t2"]
    selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    assert selector.agent.total_steps == 1


def test_select_and_score_scored_candidates_sorted_descending():
    """Scored candidates should be sorted by score (highest first)."""
    selector = _make_selector("linucb")
    candidates = ["/m/t1", "/m/t2", "/m/t3"]
    _, scored, _ = selector.select_and_score("/m/h1", "/rel/a", "/m/t1", candidates)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_keys():
    selector = _make_selector("linucb")
    s = selector.summary()
    assert "agent" in s
    assert "template_ids" in s
    assert s["template_ids"] == ["minimal", "verbose"]
