"""End-to-end experiment runner for KG link prediction."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

from src.config import get_settings
from src.data.fb15k237 import FB15k237Dataset, load_fb15k237
from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.evaluate import rank_tail_query
from src.eval.metrics import compute_all_metrics, format_metrics
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.llm_client import LLMClient
from src.models.reranker import LLMReranker
from src.models.scorer import TripleScorer
from src.prompts.renderer import PromptManager
from src.wikidata.sparql import WikidataResolver

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full link prediction experiment pipeline."""

    def __init__(
        self,
        template_id: str = "minimal",
        embedding_epochs: int = 50,
        embedding_dim: int = 128,
        num_test_queries: int = None,
        num_candidates: int = None,
        random_seed: int = None,
    ):
        settings = get_settings()
        self.template_id = template_id
        self.embedding_epochs = embedding_epochs
        self.embedding_dim = embedding_dim
        self.num_test_queries = num_test_queries or settings.sample_test_queries
        self.num_candidates = num_candidates or settings.num_candidates
        self.random_seed = random_seed or settings.random_seed
        self.results_dir = Path(settings.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Components (initialized during run)
        self.dataset: FB15k237Dataset | None = None
        self.resolver: WikidataResolver | None = None
        self.llm_client: LLMClient | None = None
        self.prompt_manager: PromptManager | None = None
        self.embedding: EmbeddingBaseline | None = None
        self.scorer: TripleScorer | None = None
        self.reranker: LLMReranker | None = None

        # Results
        self.embedding_results: list = []
        self.llm_results: list = []
        self.experiment_log: dict = {}

    def run(self) -> dict:
        """Run the full experiment pipeline."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=== Experiment started at %s ===", timestamp)
        logger.info(
            "Template: %s, Queries: %s, Candidates: %s, Seed: %s",
            self.template_id,
            self.num_test_queries,
            self.num_candidates,
            self.random_seed,
        )

        random.seed(self.random_seed)

        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset...")
        self.dataset = load_fb15k237()
        logger.info("Dataset loaded: %s", self.dataset.summary())

        # Step 2: Train embedding baseline
        logger.info("Step 2: Training embedding baseline...")
        self.embedding = EmbeddingBaseline(
            train_triples=self.dataset.train,
            valid_triples=self.dataset.valid,
            num_epochs=self.embedding_epochs,
            embedding_dim=self.embedding_dim,
            random_seed=self.random_seed,
        )
        train_summary = self.embedding.train()
        logger.info("Embedding training done: %s", train_summary)

        # Step 3: Initialize LLM components
        logger.info("Step 3: Initializing LLM components...")
        self.resolver = WikidataResolver()
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()
        template = self.prompt_manager.get(self.template_id)
        self.scorer = TripleScorer(self.llm_client, self.resolver, template)
        self.reranker = LLMReranker(self.scorer)

        # Step 4: Sample test queries
        logger.info("Step 4: Sampling %d test queries...", self.num_test_queries)
        test_queries = random.sample(
            self.dataset.test, min(self.num_test_queries, len(self.dataset.test))
        )
        all_entities = self.dataset.entities
        known_triples = self.dataset.all_triples_set

        # Step 5: Evaluate embedding baseline
        logger.info("Step 5: Evaluating embedding baseline...")
        embedding_score_fn = self.embedding.get_score_fn()
        self.embedding_results = []
        for h, r, t in test_queries:
            result = rank_tail_query(
                query=(h, r, t),
                score_fn=embedding_score_fn,
                all_entities=all_entities,
                known_triples=known_triples,
                num_candidates=self.num_candidates,
                seed=self.random_seed,
            )
            self.embedding_results.append(result)
        embedding_metrics = compute_all_metrics(self.embedding_results)
        logger.info("Embedding metrics:\n%s", format_metrics(embedding_metrics))

        # Step 6: Evaluate LLM reranker
        logger.info("Step 6: Evaluating LLM reranker...")
        self.llm_results = []
        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, self.num_candidates, seed=self.random_seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)
            result = self.reranker.rerank_tail_candidates(h, r, t, candidates)
            self.llm_results.append(result)
        llm_metrics = compute_all_metrics(self.llm_results)
        logger.info("LLM reranker metrics:\n%s", format_metrics(llm_metrics))

        # Step 7: Compile results
        elapsed = time.time() - start_time
        self.experiment_log = {
            "timestamp": timestamp,
            "config": {
                "template_id": self.template_id,
                "embedding_epochs": self.embedding_epochs,
                "embedding_dim": self.embedding_dim,
                "num_test_queries": self.num_test_queries,
                "num_candidates": self.num_candidates,
                "random_seed": self.random_seed,
            },
            "dataset_summary": self.dataset.summary(),
            "embedding_training": train_summary,
            "embedding_metrics": embedding_metrics,
            "llm_metrics": llm_metrics,
            "llm_reranker_stats": self.reranker.stats(),
            "elapsed_s": round(elapsed, 2),
        }

        # Step 8: Save results
        results_file = self.results_dir / f"experiment_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        logger.info("Results saved to %s", results_file)

        # Print comparison
        self._print_comparison(embedding_metrics, llm_metrics)

        return self.experiment_log

    def _print_comparison(self, emb_metrics: dict, llm_metrics: dict) -> None:
        """Print a side-by-side comparison of embedding vs LLM metrics."""
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<20} {'Embedding':>15} {'LLM Reranker':>15} {'Delta':>10}")
        print("-" * 60)
        for key in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            emb_val = emb_metrics.get(key, 0)
            llm_val = llm_metrics.get(key, 0)
            delta = llm_val - emb_val
            sign = "+" if delta >= 0 else ""
            print(
                f"{key:<20} {emb_val:>15.4f} {llm_val:>15.4f} {sign}{delta:>9.4f}"
            )
        print("=" * 60)
        reranker_stats = self.reranker.stats()
        llm_calls = (
            reranker_stats.get("scorer_stats", {})
            .get("llm_stats", {})
            .get("total_calls", 0)
        )
        print(f"LLM calls: {llm_calls}")
        print(f"Total time: {self.experiment_log['elapsed_s']}s")
        print()
