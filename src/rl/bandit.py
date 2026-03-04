"""Contextual bandit agents for prompt template selection."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BanditExperience:
    """A single bandit experience (context, action, reward)."""

    context: np.ndarray
    action: int  # index of chosen arm/template
    reward: float  # e.g., reciprocal rank
    action_name: str = field(default="")  # template ID for logging


class LinUCBAgent:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.

    Selects prompt templates based on query features.
    Reference: Li et al. (2010) "A Contextual-Bandit Approach to
    Personalized News Article Recommendation"
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        alpha: float = 1.0,
        arm_names: list[str] | None = None,
    ):
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.alpha = alpha  # exploration parameter
        self.arm_names = arm_names or [str(i) for i in range(n_arms)]

        # Per-arm parameters
        self.A = [np.eye(feature_dim) for _ in range(n_arms)]  # d x d
        self.b = [np.zeros(feature_dim) for _ in range(n_arms)]  # d x 1

        # History
        self.history: list[BanditExperience] = []
        self.arm_counts = [0] * n_arms
        self.arm_rewards = [0.0] * n_arms
        self.total_steps = 0

    def select_arm(self, context: np.ndarray) -> int:
        """Select an arm using LinUCB upper confidence bounds."""
        ucb_values = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            x = context
            # UCB = theta^T x + alpha * sqrt(x^T A_inv x)
            exploitation = theta @ x
            exploration = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb = exploitation + exploration
            ucb_values.append(ucb)
        return int(np.argmax(ucb_values))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update the model with observed reward."""
        x = context
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + reward * x

        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.total_steps += 1

        self.history.append(
            BanditExperience(
                context=context,
                action=action,
                reward=reward,
                action_name=self.arm_names[action],
            )
        )

    def select_and_record(
        self, context: np.ndarray, reward_fn: Callable[[int], float]
    ) -> tuple[int, float]:
        """Select arm, get reward, update model. Returns (action, reward)."""
        action = self.select_arm(context)
        reward = reward_fn(action)
        self.update(context, action, reward)
        return action, reward

    def arm_stats(self) -> list[dict]:
        """Per-arm statistics."""
        stats = []
        for i in range(self.n_arms):
            avg_reward = self.arm_rewards[i] / max(1, self.arm_counts[i])
            stats.append(
                {
                    "arm": i,
                    "name": self.arm_names[i],
                    "count": self.arm_counts[i],
                    "total_reward": round(self.arm_rewards[i], 4),
                    "avg_reward": round(avg_reward, 4),
                }
            )
        return stats

    def summary(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "alpha": self.alpha,
            "n_arms": self.n_arms,
            "feature_dim": self.feature_dim,
            "arm_stats": self.arm_stats(),
        }


class EpsilonGreedyAgent:
    """
    Simple epsilon-greedy contextual bandit baseline.

    Maintains per-arm running average reward (ignores context).
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        arm_names: list[str] | None = None,
    ):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_names = arm_names or [str(i) for i in range(n_arms)]

        self.arm_counts = [0] * n_arms
        self.arm_rewards = [0.0] * n_arms
        self.total_steps = 0
        self.history: list[BanditExperience] = []
        self._rng = np.random.RandomState(42)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        """Epsilon-greedy selection (context ignored)."""
        if self._rng.random() < self.epsilon or self.total_steps < self.n_arms:
            return int(self._rng.randint(self.n_arms))
        avg = [
            self.arm_rewards[i] / max(1, self.arm_counts[i])
            for i in range(self.n_arms)
        ]
        return int(np.argmax(avg))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        self.total_steps += 1
        self.history.append(
            BanditExperience(
                context=context,
                action=action,
                reward=reward,
                action_name=self.arm_names[action],
            )
        )

    def arm_stats(self) -> list[dict]:
        stats = []
        for i in range(self.n_arms):
            avg_reward = self.arm_rewards[i] / max(1, self.arm_counts[i])
            stats.append(
                {
                    "arm": i,
                    "name": self.arm_names[i],
                    "count": self.arm_counts[i],
                    "total_reward": round(self.arm_rewards[i], 4),
                    "avg_reward": round(avg_reward, 4),
                }
            )
        return stats

    def summary(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "n_arms": self.n_arms,
            "arm_stats": self.arm_stats(),
        }
