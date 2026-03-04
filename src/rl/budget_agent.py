"""RL budget agent that decides whether to use LLM reranker or embedding-only."""

from __future__ import annotations

import logging

from src.rl.bandit import EpsilonGreedyAgent, LinUCBAgent
from src.rl.features import QueryFeatureExtractor

logger = logging.getLogger(__name__)


class BudgetAgent:
    """
    RL agent that decides whether to use the LLM reranker or fallback to
    embedding-only scoring for each query, under a fixed LLM call budget.

    Arms:
      0 = embedding-only (free)
      1 = LLM reranker (costs budget)

    The agent learns which queries benefit most from LLM reranking
    and allocates the limited budget accordingly.
    """

    ARM_EMBEDDING = 0
    ARM_LLM = 1
    ARM_NAMES = ["embedding_only", "llm_reranker"]

    def __init__(
        self,
        feature_extractor: QueryFeatureExtractor,
        total_budget: int = 100,
        agent_type: str = "linucb",
        alpha: float = 1.0,
        epsilon: float = 0.1,
        cost_per_llm_query: int = 1,
    ):
        self.feature_extractor = feature_extractor
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.cost_per_llm_query = cost_per_llm_query

        n_arms = 2
        if agent_type == "linucb":
            self.agent = LinUCBAgent(
                n_arms=n_arms,
                feature_dim=feature_extractor.feature_dim,
                alpha=alpha,
                arm_names=self.ARM_NAMES,
            )
        elif agent_type == "epsilon_greedy":
            self.agent = EpsilonGreedyAgent(
                n_arms=n_arms,
                epsilon=epsilon,
                arm_names=self.ARM_NAMES,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Tracking
        self.decisions: list[dict] = []
        self.total_queries = 0
        self.llm_queries = 0
        self.embedding_queries = 0

    def decide(self, h: str, r: str, t: str) -> int:
        """
        Decide whether to use embedding (0) or LLM (1) for this query.
        If budget is insufficient for an LLM call, always returns embedding.
        """
        if self.remaining_budget < self.cost_per_llm_query:
            return self.ARM_EMBEDDING

        context = self.feature_extractor.extract(h, r, t)
        return self.agent.select_arm(context)

    def record_decision(
        self,
        h: str,
        r: str,
        t: str,
        action: int,
        reward: float,
        embedding_rank: int | None = None,
        llm_rank: int | None = None,
    ):
        """Record the outcome and update the agent."""
        context = self.feature_extractor.extract(h, r, t)
        self.agent.update(context, action, reward)

        if action == self.ARM_LLM:
            self.remaining_budget -= self.cost_per_llm_query
            self.llm_queries += 1
        else:
            self.embedding_queries += 1

        self.total_queries += 1

        self.decisions.append(
            {
                "query": (h, r, t),
                "action": self.ARM_NAMES[action],
                "reward": reward,
                "embedding_rank": embedding_rank,
                "llm_rank": llm_rank,
                "remaining_budget": self.remaining_budget,
            }
        )

    @property
    def budget_utilization(self) -> float:
        """Fraction of budget used."""
        used = self.total_budget - self.remaining_budget
        return used / max(1, self.total_budget)

    @property
    def llm_fraction(self) -> float:
        """Fraction of queries sent to LLM."""
        return self.llm_queries / max(1, self.total_queries)

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "remaining_budget": self.remaining_budget,
            "budget_utilization": round(self.budget_utilization, 4),
            "total_queries": self.total_queries,
            "llm_queries": self.llm_queries,
            "embedding_queries": self.embedding_queries,
            "llm_fraction": round(self.llm_fraction, 4),
            "agent_summary": self.agent.summary(),
        }

    def reset_budget(self, new_budget: int | None = None):
        """Reset budget (e.g., for a new evaluation run)."""
        self.remaining_budget = new_budget or self.total_budget
        self.decisions.clear()
        self.total_queries = 0
        self.llm_queries = 0
        self.embedding_queries = 0
