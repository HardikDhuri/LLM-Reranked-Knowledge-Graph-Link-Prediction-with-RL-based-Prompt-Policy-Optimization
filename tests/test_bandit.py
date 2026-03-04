"""Tests for src/rl/bandit.py — LinUCBAgent, EpsilonGreedyAgent, BanditExperience."""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.bandit import BanditExperience, EpsilonGreedyAgent, LinUCBAgent

# ---------------------------------------------------------------------------
# BanditExperience
# ---------------------------------------------------------------------------


def test_bandit_experience_dataclass():
    ctx = np.zeros(4, dtype=np.float32)
    exp = BanditExperience(context=ctx, action=2, reward=0.5, action_name="tmpl_a")
    assert exp.action == 2
    assert exp.reward == pytest.approx(0.5)
    assert exp.action_name == "tmpl_a"
    assert np.array_equal(exp.context, ctx)


def test_bandit_experience_default_action_name():
    ctx = np.ones(3, dtype=np.float32)
    exp = BanditExperience(context=ctx, action=0, reward=1.0)
    assert exp.action_name == ""


# ---------------------------------------------------------------------------
# LinUCBAgent — select_arm
# ---------------------------------------------------------------------------


def _make_linucb(n_arms: int = 3, feature_dim: int = 4, alpha: float = 1.0):
    names = [f"arm_{i}" for i in range(n_arms)]
    return LinUCBAgent(
        n_arms=n_arms, feature_dim=feature_dim, alpha=alpha, arm_names=names
    )


def test_linucb_select_arm_returns_valid_index():
    agent = _make_linucb(n_arms=4, feature_dim=8)
    ctx = np.random.rand(8).astype(np.float32)
    arm = agent.select_arm(ctx)
    assert 0 <= arm < 4


def test_linucb_select_arm_always_valid_over_many_calls():
    agent = _make_linucb(n_arms=3, feature_dim=5)
    rng = np.random.RandomState(0)
    for _ in range(50):
        ctx = rng.rand(5).astype(np.float32)
        arm = agent.select_arm(ctx)
        assert 0 <= arm < 3


# ---------------------------------------------------------------------------
# LinUCBAgent — update
# ---------------------------------------------------------------------------


def test_linucb_update_changes_A_matrix():
    agent = _make_linucb(n_arms=2, feature_dim=3)
    A_before = agent.A[0].copy()
    ctx = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    agent.update(ctx, action=0, reward=0.8)
    assert not np.allclose(agent.A[0], A_before)


def test_linucb_update_changes_b_vector():
    agent = _make_linucb(n_arms=2, feature_dim=3)
    b_before = agent.b[0].copy()
    ctx = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    agent.update(ctx, action=0, reward=0.8)
    assert not np.allclose(agent.b[0], b_before)


def test_linucb_update_increments_total_steps():
    agent = _make_linucb()
    ctx = np.ones(4, dtype=np.float32)
    agent.update(ctx, action=0, reward=1.0)
    assert agent.total_steps == 1
    agent.update(ctx, action=1, reward=0.5)
    assert agent.total_steps == 2


def test_linucb_update_records_history():
    agent = _make_linucb()
    ctx = np.ones(4, dtype=np.float32)
    agent.update(ctx, action=0, reward=1.0)
    assert len(agent.history) == 1
    assert agent.history[0].action == 0
    assert agent.history[0].reward == pytest.approx(1.0)


def test_linucb_prefers_high_reward_arm_after_training():
    """After many updates rewarding arm 0, it should be selected consistently."""
    agent = LinUCBAgent(n_arms=3, feature_dim=2, alpha=0.01)
    ctx = np.array([1.0, 0.0], dtype=np.float32)
    # Arm 0 always gets reward 1.0, others get 0.0
    for _ in range(30):
        agent.update(ctx, action=0, reward=1.0)
        agent.update(ctx, action=1, reward=0.0)
        agent.update(ctx, action=2, reward=0.0)
    chosen = agent.select_arm(ctx)
    assert chosen == 0


# ---------------------------------------------------------------------------
# LinUCBAgent — arm_stats & summary
# ---------------------------------------------------------------------------


def test_linucb_arm_stats_structure():
    agent = _make_linucb(n_arms=2)
    ctx = np.ones(4, dtype=np.float32)
    agent.update(ctx, action=0, reward=0.5)
    stats = agent.arm_stats()
    assert len(stats) == 2
    for s in stats:
        assert "arm" in s
        assert "name" in s
        assert "count" in s
        assert "total_reward" in s
        assert "avg_reward" in s


def test_linucb_arm_stats_counts_correctly():
    agent = _make_linucb(n_arms=2)
    ctx = np.ones(4, dtype=np.float32)
    agent.update(ctx, action=0, reward=0.5)
    agent.update(ctx, action=0, reward=0.5)
    stats = agent.arm_stats()
    assert stats[0]["count"] == 2
    assert stats[1]["count"] == 0


def test_linucb_summary_keys():
    agent = _make_linucb()
    s = agent.summary()
    assert "total_steps" in s
    assert "alpha" in s
    assert "n_arms" in s
    assert "feature_dim" in s
    assert "arm_stats" in s


# ---------------------------------------------------------------------------
# EpsilonGreedyAgent — select_arm
# ---------------------------------------------------------------------------


def _make_eps(n_arms: int = 3, epsilon: float = 0.1):
    names = [f"arm_{i}" for i in range(n_arms)]
    return EpsilonGreedyAgent(n_arms=n_arms, epsilon=epsilon, arm_names=names)


def test_eps_select_arm_returns_valid_index():
    agent = _make_eps(n_arms=4)
    arm = agent.select_arm(None)
    assert 0 <= arm < 4


def test_eps_greedy_zero_epsilon_picks_best_arm():
    """With epsilon=0, after enough updates, always pick best arm."""
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, arm_names=["a", "b", "c"])
    ctx = np.zeros(2, dtype=np.float32)
    # Give arm 2 high reward
    for _ in range(10):
        agent.update(ctx, action=0, reward=0.1)
        agent.update(ctx, action=1, reward=0.2)
        agent.update(ctx, action=2, reward=0.9)
    # After warm-up, should always pick arm 2
    chosen = agent.select_arm(ctx)
    assert chosen == 2


def test_eps_greedy_epsilon_one_explores():
    """With epsilon=1, selection is random (never purely greedy)."""
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=1.0)
    ctx = np.zeros(2, dtype=np.float32)
    # Give arm 0 the best reward to set up greedy baseline
    for _ in range(10):
        agent.update(ctx, action=0, reward=1.0)
        agent.update(ctx, action=1, reward=0.0)
        agent.update(ctx, action=2, reward=0.0)
    # With epsilon=1 all choices should be random; run many times and
    # check that not all choices are arm 0
    choices = {agent.select_arm(ctx) for _ in range(50)}
    assert len(choices) > 1  # must explore beyond best arm


# ---------------------------------------------------------------------------
# EpsilonGreedyAgent — update, arm_stats, summary
# ---------------------------------------------------------------------------


def test_eps_update_increments_steps():
    agent = _make_eps()
    ctx = np.zeros(2, dtype=np.float32)
    agent.update(ctx, action=0, reward=0.5)
    assert agent.total_steps == 1


def test_eps_arm_stats_structure():
    agent = _make_eps(n_arms=2)
    ctx = np.zeros(2, dtype=np.float32)
    agent.update(ctx, action=0, reward=1.0)
    stats = agent.arm_stats()
    assert len(stats) == 2
    for s in stats:
        assert "arm" in s
        assert "name" in s
        assert "count" in s
        assert "total_reward" in s
        assert "avg_reward" in s


def test_eps_summary_keys():
    agent = _make_eps()
    s = agent.summary()
    assert "total_steps" in s
    assert "epsilon" in s
    assert "n_arms" in s
    assert "arm_stats" in s
