"""Unit tests for src/rl/budget_agent.py — BudgetAgent."""

from __future__ import annotations

import pytest

from src.rl.budget_agent import BudgetAgent
from src.rl.features import QueryFeatureExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAIN = [
    ("e0", "r0", "e1"),
    ("e1", "r0", "e2"),
    ("e2", "r0", "e3"),
    ("e3", "r0", "e4"),
    ("e0", "r1", "e2"),
]


def _make_extractor() -> QueryFeatureExtractor:
    return QueryFeatureExtractor.from_triples(_TRAIN)


def _make_agent(
    total_budget: int = 10,
    agent_type: str = "linucb",
    alpha: float = 1.0,
    epsilon: float = 0.1,
) -> BudgetAgent:
    extractor = _make_extractor()
    return BudgetAgent(
        feature_extractor=extractor,
        total_budget=total_budget,
        agent_type=agent_type,
        alpha=alpha,
        epsilon=epsilon,
    )


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_default_budget_and_counters():
    agent = _make_agent(total_budget=20)
    assert agent.total_budget == 20
    assert agent.remaining_budget == 20
    assert agent.total_queries == 0
    assert agent.llm_queries == 0
    assert agent.embedding_queries == 0
    assert agent.decisions == []


def test_init_linucb_agent_type():
    from src.rl.bandit import LinUCBAgent

    agent = _make_agent(agent_type="linucb")
    assert isinstance(agent.agent, LinUCBAgent)


def test_init_epsilon_greedy_agent_type():
    from src.rl.bandit import EpsilonGreedyAgent

    agent = _make_agent(agent_type="epsilon_greedy")
    assert isinstance(agent.agent, EpsilonGreedyAgent)


def test_init_unknown_agent_type_raises():
    with pytest.raises(ValueError, match="Unknown agent type"):
        _make_agent(agent_type="unknown")


def test_arm_constants():
    assert BudgetAgent.ARM_EMBEDDING == 0
    assert BudgetAgent.ARM_LLM == 1
    assert len(BudgetAgent.ARM_NAMES) == 2


# ---------------------------------------------------------------------------
# decide() tests
# ---------------------------------------------------------------------------


def test_decide_returns_0_or_1():
    agent = _make_agent(total_budget=10)
    action = agent.decide("e0", "r0", "e1")
    assert action in (0, 1)


def test_decide_returns_embedding_when_budget_exhausted():
    agent = _make_agent(total_budget=0)
    for _ in range(5):
        action = agent.decide("e0", "r0", "e1")
        assert action == BudgetAgent.ARM_EMBEDDING


def test_decide_returns_embedding_when_remaining_budget_zero():
    agent = _make_agent(total_budget=5)
    agent.remaining_budget = 0
    action = agent.decide("e0", "r0", "e1")
    assert action == BudgetAgent.ARM_EMBEDDING


def test_decide_returns_embedding_when_budget_less_than_cost():
    agent = _make_agent(total_budget=5)
    agent.remaining_budget = 0  # cost_per_llm_query=1, so 0 < 1
    action = agent.decide("e0", "r0", "e1")
    assert action == BudgetAgent.ARM_EMBEDDING


def test_decide_returns_embedding_when_cost_exceeds_remaining():
    """Budget is positive but less than cost_per_llm_query."""
    extractor = _make_extractor()
    agent = BudgetAgent(
        feature_extractor=extractor,
        total_budget=10,
        cost_per_llm_query=3,
    )
    agent.remaining_budget = 2  # 2 < cost_per_llm_query=3
    action = agent.decide("e0", "r0", "e1")
    assert action == BudgetAgent.ARM_EMBEDDING


# ---------------------------------------------------------------------------
# record_decision() tests
# ---------------------------------------------------------------------------


def test_record_decision_llm_decrements_budget():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    assert agent.remaining_budget == 9


def test_record_decision_embedding_does_not_decrement_budget():
    agent = _make_agent(total_budget=10)
    agent.record_decision(
        "e0", "r0", "e1", action=BudgetAgent.ARM_EMBEDDING, reward=0.3
    )
    assert agent.remaining_budget == 10


def test_record_decision_increments_llm_counter():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    assert agent.llm_queries == 1
    assert agent.embedding_queries == 0
    assert agent.total_queries == 1


def test_record_decision_increments_embedding_counter():
    agent = _make_agent(total_budget=10)
    agent.record_decision(
        "e0", "r0", "e1", action=BudgetAgent.ARM_EMBEDDING, reward=0.3
    )
    assert agent.embedding_queries == 1
    assert agent.llm_queries == 0
    assert agent.total_queries == 1


def test_record_decision_appends_to_decisions():
    agent = _make_agent(total_budget=10)
    agent.record_decision(
        "e0",
        "r0",
        "e1",
        action=BudgetAgent.ARM_LLM,
        reward=0.5,
        embedding_rank=3,
        llm_rank=1,
    )
    assert len(agent.decisions) == 1
    d = agent.decisions[0]
    assert d["query"] == ("e0", "r0", "e1")
    assert d["action"] == "llm_reranker"
    assert d["reward"] == pytest.approx(0.5)
    assert d["embedding_rank"] == 3
    assert d["llm_rank"] == 1


def test_record_multiple_decisions():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=1.0)
    agent.record_decision(
        "e1", "r0", "e2", action=BudgetAgent.ARM_EMBEDDING, reward=0.5
    )
    agent.record_decision("e2", "r0", "e3", action=BudgetAgent.ARM_LLM, reward=0.25)
    assert agent.llm_queries == 2
    assert agent.embedding_queries == 1
    assert agent.total_queries == 3
    assert agent.remaining_budget == 8  # 10 - 2 LLM queries


# ---------------------------------------------------------------------------
# budget_utilization and llm_fraction properties
# ---------------------------------------------------------------------------


def test_budget_utilization_initial():
    agent = _make_agent(total_budget=10)
    assert agent.budget_utilization == pytest.approx(0.0)


def test_budget_utilization_after_llm_calls():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    agent.record_decision("e1", "r0", "e2", action=BudgetAgent.ARM_LLM, reward=0.5)
    assert agent.budget_utilization == pytest.approx(0.2)  # 2/10


def test_budget_utilization_no_budget():
    agent = _make_agent(total_budget=0)
    # With 0 budget, no LLM queries can be made; utilization = 0/max(1,0) = 0
    assert agent.budget_utilization == pytest.approx(0.0)


def test_llm_fraction_initial():
    agent = _make_agent(total_budget=10)
    assert agent.llm_fraction == pytest.approx(0.0)


def test_llm_fraction_after_queries():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    agent.record_decision(
        "e1", "r0", "e2", action=BudgetAgent.ARM_EMBEDDING, reward=0.5
    )
    agent.record_decision(
        "e2", "r0", "e3", action=BudgetAgent.ARM_EMBEDDING, reward=0.5
    )
    agent.record_decision("e3", "r0", "e4", action=BudgetAgent.ARM_LLM, reward=0.5)
    assert agent.llm_fraction == pytest.approx(0.5)  # 2/4


# ---------------------------------------------------------------------------
# reset_budget() tests
# ---------------------------------------------------------------------------


def test_reset_budget_clears_state():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    agent.record_decision(
        "e1", "r0", "e2", action=BudgetAgent.ARM_EMBEDDING, reward=0.3
    )
    agent.reset_budget()
    assert agent.remaining_budget == 10
    assert agent.total_queries == 0
    assert agent.llm_queries == 0
    assert agent.embedding_queries == 0
    assert agent.decisions == []


def test_reset_budget_with_new_budget():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    agent.reset_budget(new_budget=25)
    assert agent.remaining_budget == 25


# ---------------------------------------------------------------------------
# summary() tests
# ---------------------------------------------------------------------------


def test_summary_returns_expected_keys():
    agent = _make_agent(total_budget=10)
    s = agent.summary()
    expected_keys = [
        "total_budget",
        "remaining_budget",
        "budget_utilization",
        "total_queries",
        "llm_queries",
        "embedding_queries",
        "llm_fraction",
        "agent_summary",
    ]
    for key in expected_keys:
        assert key in s, f"Missing key: {key}"


def test_summary_values_are_correct():
    agent = _make_agent(total_budget=10)
    agent.record_decision("e0", "r0", "e1", action=BudgetAgent.ARM_LLM, reward=0.5)
    s = agent.summary()
    assert s["total_budget"] == 10
    assert s["remaining_budget"] == 9
    assert s["total_queries"] == 1
    assert s["llm_queries"] == 1
    assert s["embedding_queries"] == 0
    assert s["budget_utilization"] == pytest.approx(0.1)
    assert s["llm_fraction"] == pytest.approx(1.0)
    assert "agent_summary" in s
