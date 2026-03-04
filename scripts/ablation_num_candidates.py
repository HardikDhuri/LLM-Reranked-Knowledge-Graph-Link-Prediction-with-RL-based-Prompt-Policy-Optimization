#!/usr/bin/env python
"""Ablation study: sweep num_candidates and measure embedding + LLM reranker quality.

Usage::

    python -m scripts.ablation_num_candidates --candidates 5,10,25 --num-queries 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.data.fb15k237 import load_fb15k237
from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import RankingResult, compute_all_metrics
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.llm_client import LLMClient
from src.models.reranker import LLMReranker
from src.models.scorer import TripleScorer
from src.prompts.renderer import PromptManager
from src.wikidata.sparql import WikidataResolver

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_sweep(
    candidate_counts: list[int],
    template_id: str,
    num_queries: int,
    random_seed: int | None = None,
) -> list[dict]:
    """Run candidate count sweep and return results for each K."""
    import random

    dataset = load_fb15k237()
    all_entities = dataset.entities
    known_triples = dataset.all_triples_set

    # Train a single embedding for reuse across all candidate counts
    logger.info("Training embedding baseline...")
    embedding = EmbeddingBaseline(
        train_triples=dataset.train,
        valid_triples=dataset.valid,
        num_epochs=50,
        embedding_dim=128,
        random_seed=random_seed,
    )
    embedding.train()
    score_fn = embedding.get_score_fn()

    # Initialize LLM components
    resolver = WikidataResolver()
    llm_client = LLMClient()
    prompt_manager = PromptManager()
    template = prompt_manager.get(template_id)
    scorer = TripleScorer(llm_client, resolver, template)
    reranker = LLMReranker(scorer)

    random.seed(random_seed)
    test_queries = random.sample(dataset.test, min(num_queries, len(dataset.test)))

    results = []
    for k in candidate_counts:
        logger.info(f"Evaluating with num_candidates={k}")
        emb_results = []
        llm_results = []

        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, k, seed=random_seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)

            # Embedding baseline
            scored = [(tail, score_fn(h, r, tail)) for tail in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [tail for tail, _ in scored]
            rank = ranked.index(t) + 1 if t in ranked else len(ranked) + 1
            emb_results.append(
                RankingResult(
                    query=(h, r, t),
                    true_rank=rank,
                    num_candidates=len(candidates),
                    scored_candidates=scored,
                )
            )

            # LLM reranker
            llm_result = reranker.rerank_tail_candidates(h, r, t, candidates)
            llm_results.append(llm_result)

        emb_metrics = compute_all_metrics(emb_results)
        llm_metrics = compute_all_metrics(llm_results)
        reranker_stats = reranker.stats()
        llm_calls = (
            reranker_stats.get("scorer_stats", {})
            .get("llm_stats", {})
            .get("total_calls", 0)
        )

        results.append(
            {
                "num_candidates": k,
                "embedding_metrics": emb_metrics,
                "llm_metrics": llm_metrics,
                "llm_calls": llm_calls,
            }
        )
        logger.info(
            f"k={k}: Emb MRR={emb_metrics.get('MRR', 0):.4f}, "
            f"LLM MRR={llm_metrics.get('MRR', 0):.4f}, "
            f"LLM calls={llm_calls}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study over number of candidates"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="5,10,25,50",
        help="Comma-separated list of candidate counts",
    )
    parser.add_argument(
        "--template", type=str, default="minimal", help="Prompt template ID"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of test queries to evaluate",
    )
    args = parser.parse_args()

    candidate_counts = [int(c.strip()) for c in args.candidates.split(",")]
    settings = get_settings()

    logger.info(f"Ablation sweep over candidate counts: {candidate_counts}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = run_sweep(
        candidate_counts=candidate_counts,
        template_id=args.template,
        num_queries=args.num_queries,
        random_seed=settings.random_seed,
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION: NUM CANDIDATES")
    print("=" * 80)
    print(
        f"{'K':>6} {'Emb MRR':>10} {'LLM MRR':>10} "
        f"{'Emb H@10':>10} {'LLM H@10':>10} {'LLM Calls':>10}"
    )
    print("-" * 80)
    for row in results:
        em = row["embedding_metrics"]
        lm = row["llm_metrics"]
        print(
            f"{row['num_candidates']:>6} {em.get('MRR', 0):>10.4f} "
            f"{lm.get('MRR', 0):>10.4f} {em.get('Hits@10', 0):>10.4f} "
            f"{lm.get('Hits@10', 0):>10.4f} {row['llm_calls']:>10}"
        )
    print("=" * 80)

    # Save results
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"ablation_num_candidates_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": {
            "candidate_counts": candidate_counts,
            "template": args.template,
            "num_queries": args.num_queries,
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
