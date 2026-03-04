"""Budget-constrained experiment comparing embedding-only, LLM, and RL strategies."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

from src.config import get_settings
from src.data.fb15k237 import load_fb15k237
from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import RankingResult, compute_all_metrics
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.llm_client import LLMClient
from src.models.reranker import LLMReranker
from src.models.scorer import TripleScorer
from src.prompts.renderer import PromptManager
from src.rl.budget_agent import BudgetAgent
from src.rl.features import QueryFeatureExtractor
from src.wikidata.sparql import WikidataResolver

logger = logging.getLogger(__name__)


class BudgetExperiment:
    """
    Runs a budget-constrained experiment comparing:
    1. Embedding-only baseline (all queries)
    2. LLM-only reranker (all queries, if budget allows)
    3. RL budget agent (selectively uses LLM)
    """

    def __init__(
        self,
        template_id: str = "minimal",
        total_budget: int = 50,
        agent_type: str = "linucb",
        alpha: float = 1.0,
        embedding_epochs: int = 50,
        embedding_dim: int = 128,
        num_test_queries: int | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
    ):
        settings = get_settings()
        self.template_id = template_id
        self.total_budget = total_budget
        self.agent_type = agent_type
        self.alpha = alpha
        self.embedding_epochs = embedding_epochs
        self.embedding_dim = embedding_dim
        self.num_test_queries = num_test_queries or settings.sample_test_queries
        self.num_candidates = num_candidates or settings.num_candidates
        self.random_seed = random_seed or settings.random_seed
        self.results_dir = Path(settings.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """Run the full budget experiment."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random.seed(self.random_seed)

        logger.info("=== Budget Experiment started ===")
        logger.info(
            f"Budget: {self.total_budget}, Agent: {self.agent_type}, "
            f"Template: {self.template_id}"
        )

        # Step 1: Load dataset
        dataset = load_fb15k237()
        all_entities = dataset.entities
        known_triples = dataset.all_triples_set

        # Step 2: Train embedding
        embedding = EmbeddingBaseline(
            train_triples=dataset.train,
            valid_triples=dataset.valid,
            num_epochs=self.embedding_epochs,
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
        )
        embedding.train()
        embedding_score_fn = embedding.get_score_fn()

        # Step 3: Initialize LLM components
        resolver = WikidataResolver()
        llm_client = LLMClient()
        prompt_manager = PromptManager()
        template = prompt_manager.get(self.template_id)
        scorer = TripleScorer(llm_client, resolver, template)
        reranker = LLMReranker(scorer)

        # Step 4: Build feature extractor + budget agent
        feature_extractor = QueryFeatureExtractor.from_triples(dataset.train)
        budget_agent = BudgetAgent(
            feature_extractor=feature_extractor,
            total_budget=self.total_budget,
            agent_type=self.agent_type,
            alpha=self.alpha,
        )

        # Step 5: Sample test queries
        test_queries = random.sample(
            dataset.test, min(self.num_test_queries, len(dataset.test))
        )

        # Step 6: Evaluate all strategies
        embedding_results = []
        rl_budget_results = []

        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, self.num_candidates, seed=self.random_seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)

            # A) Embedding-only
            emb_scored = [(tail, embedding_score_fn(h, r, tail)) for tail in candidates]
            emb_scored.sort(key=lambda x: x[1], reverse=True)
            emb_ranked = [tail for tail, _ in emb_scored]
            emb_rank = (
                emb_ranked.index(t) + 1 if t in emb_ranked else len(emb_ranked) + 1
            )
            embedding_results.append(
                RankingResult(
                    query=(h, r, t),
                    true_rank=emb_rank,
                    num_candidates=len(candidates),
                    scored_candidates=emb_scored,
                )
            )

            # B) RL budget agent decision
            action = budget_agent.decide(h, r, t)
            if action == BudgetAgent.ARM_LLM:
                llm_result = reranker.rerank_tail_candidates(h, r, t, candidates)
                rl_rank = llm_result.true_rank
                rl_scored = llm_result.scored_candidates
            else:
                rl_rank = emb_rank
                rl_scored = emb_scored

            reward = 1.0 / rl_rank
            budget_agent.record_decision(
                h,
                r,
                t,
                action,
                reward,
                embedding_rank=emb_rank,
                llm_rank=rl_rank if action == BudgetAgent.ARM_LLM else None,
            )

            rl_budget_results.append(
                RankingResult(
                    query=(h, r, t),
                    true_rank=rl_rank,
                    num_candidates=len(candidates),
                    scored_candidates=rl_scored,
                )
            )

        # Step 7: Compute metrics
        emb_metrics = compute_all_metrics(embedding_results)
        rl_metrics = compute_all_metrics(rl_budget_results)

        elapsed = time.time() - start_time

        results = {
            "timestamp": timestamp,
            "config": {
                "template_id": self.template_id,
                "total_budget": self.total_budget,
                "agent_type": self.agent_type,
                "alpha": self.alpha,
                "num_test_queries": self.num_test_queries,
                "num_candidates": self.num_candidates,
            },
            "embedding_metrics": emb_metrics,
            "rl_budget_metrics": rl_metrics,
            "budget_agent_summary": budget_agent.summary(),
            "elapsed_s": round(elapsed, 2),
        }

        # Save results
        results_file = self.results_dir / f"budget_experiment_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")

        # Print comparison
        self._print_comparison(emb_metrics, rl_metrics, budget_agent)

        return results

    def _print_comparison(self, emb: dict, rl: dict, agent: BudgetAgent):
        print("\n" + "=" * 65)
        print("BUDGET EXPERIMENT RESULTS")
        print("=" * 65)
        print(f"{'Metric':<20} {'Embedding-Only':>15} {'RL Budget':>15} {'Delta':>10}")
        print("-" * 65)
        for key in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            e = emb.get(key, 0)
            r = rl.get(key, 0)
            d = r - e
            sign = "+" if d >= 0 else ""
            print(f"{key:<20} {e:>15.4f} {r:>15.4f} {sign}{d:>9.4f}")
        print("-" * 65)
        used = agent.total_budget - agent.remaining_budget
        print(f"Budget used: {used}/{agent.total_budget}")
        print(
            f"LLM queries: {agent.llm_queries}/{agent.total_queries} "
            f"({agent.llm_fraction:.1%})"
        )
        print(f"Embedding queries: {agent.embedding_queries}/{agent.total_queries}")
        print("=" * 65)
