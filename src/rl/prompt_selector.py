"""RL-based prompt template selector using contextual bandits."""

from __future__ import annotations

import logging

from src.models.llm_client import LLMClient
from src.models.scorer import TripleScorer
from src.prompts.renderer import PromptManager
from src.rl.bandit import EpsilonGreedyAgent, LinUCBAgent
from src.rl.features import QueryFeatureExtractor
from src.wikidata.sparql import WikidataResolver

logger = logging.getLogger(__name__)


class RLPromptSelector:
    """
    Uses a contextual bandit to select the best prompt template per query.

    Reward = reciprocal rank (1/rank) after reranking with the selected template.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        resolver: WikidataResolver,
        prompt_manager: PromptManager,
        feature_extractor: QueryFeatureExtractor,
        agent_type: str = "linucb",
        alpha: float = 1.0,
        epsilon: float = 0.1,
    ):
        self.llm_client = llm_client
        self.resolver = resolver
        self.prompt_manager = prompt_manager
        self.feature_extractor = feature_extractor

        self.template_ids = prompt_manager.list_ids()
        n_arms = len(self.template_ids)

        if agent_type == "linucb":
            self.agent: LinUCBAgent | EpsilonGreedyAgent = LinUCBAgent(
                n_arms=n_arms,
                feature_dim=feature_extractor.feature_dim,
                alpha=alpha,
                arm_names=self.template_ids,
            )
        elif agent_type == "epsilon_greedy":
            self.agent = EpsilonGreedyAgent(
                n_arms=n_arms,
                epsilon=epsilon,
                arm_names=self.template_ids,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Pre-build scorers for each template
        self.scorers: dict[str, TripleScorer] = {}
        for tid in self.template_ids:
            template = prompt_manager.get(tid)
            self.scorers[tid] = TripleScorer(llm_client, resolver, template)

    def select_template(self, h: str, r: str, t: str) -> str:
        """Select a template for the given query using the bandit agent."""
        context = self.feature_extractor.extract(h, r, t)
        arm_idx = self.agent.select_arm(context)
        return self.template_ids[arm_idx]

    def select_and_score(
        self,
        h: str,
        r: str,
        true_t: str,
        candidates: list[str],
    ) -> tuple[str, list[tuple[str, float]], int]:
        """
        Select a template, score all candidates, compute rank, update agent.

        Returns: (selected_template_id, scored_candidates, true_rank)
        """
        context = self.feature_extractor.extract(h, r, true_t)
        arm_idx = self.agent.select_arm(context)
        template_id = self.template_ids[arm_idx]
        scorer = self.scorers[template_id]

        # Score each candidate
        scored = []
        for tail in candidates:
            score = scorer.score_triple(h, r, tail)
            scored.append((tail, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Find true rank
        ranked_entities = [ent for ent, _ in scored]
        if true_t in ranked_entities:
            true_rank = ranked_entities.index(true_t) + 1
        else:
            true_rank = len(ranked_entities) + 1

        # Reward = reciprocal rank
        reward = 1.0 / true_rank
        self.agent.update(context, arm_idx, reward)

        logger.debug(
            "Query (%s, %s, %s): template=%s, rank=%d, reward=%.4f",
            h,
            r,
            true_t,
            template_id,
            true_rank,
            reward,
        )

        return template_id, scored, true_rank

    def summary(self) -> dict:
        return {
            "agent": self.agent.summary(),
            "template_ids": self.template_ids,
        }
