#!/usr/bin/env python
"""Compare all prompt templates on the same test queries.

Runs the LLM reranker experiment for each available template and prints
a side-by-side comparison table. Saves aggregated results to
``results/template_comparison_<timestamp>.json``.

Usage::

    python -m scripts.compare_templates --num-queries 5 --num-candidates 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.data.fb15k237 import load_fb15k237
from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import compute_all_metrics, format_metrics
from src.models.llm_client import LLMClient
from src.models.reranker import LLMReranker
from src.models.scorer import TripleScorer
from src.prompts.renderer import PromptManager
from src.wikidata.sparql import WikidataResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all prompt templates on the same test queries."
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of test queries to evaluate (default: from config)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of candidates per query (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    settings = get_settings()
    num_queries = args.num_queries or settings.sample_test_queries
    num_candidates = args.num_candidates or settings.num_candidates
    seed = args.seed or settings.random_seed

    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset once
    logger.info("Loading FB15k-237 dataset...")
    dataset = load_fb15k237()
    logger.info("Dataset loaded: %s", dataset.summary())

    # Sample the same test queries for all templates
    random.seed(seed)
    test_queries = random.sample(
        dataset.test, min(num_queries, len(dataset.test))
    )
    all_entities = dataset.entities
    known_triples = dataset.all_triples_set

    # Shared LLM / resolver components
    resolver = WikidataResolver()
    llm_client = LLMClient()
    registry = PromptManager()
    template_ids = registry.list_ids()

    logger.info("Comparing %d templates: %s", len(template_ids), template_ids)

    comparison: dict = {
        "timestamp": timestamp,
        "config": {
            "num_test_queries": num_queries,
            "num_candidates": num_candidates,
            "random_seed": seed,
        },
        "dataset_summary": dataset.summary(),
        "templates": {},
    }

    for tid in template_ids:
        logger.info("--- Template: %s ---", tid)
        template = registry.get(tid)
        scorer = TripleScorer(llm_client, resolver, template)
        reranker = LLMReranker(scorer)

        t0 = time.time()
        results = []
        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, num_candidates, seed=seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)
            result = reranker.rerank_tail_candidates(h, r, t, candidates)
            results.append(result)

        metrics = compute_all_metrics(results)
        elapsed = round(time.time() - t0, 2)
        logger.info("Template %s metrics:\n%s", tid, format_metrics(metrics))

        comparison["templates"][tid] = {
            "metrics": metrics,
            "reranker_stats": reranker.stats(),
            "elapsed_s": elapsed,
        }

    # Print comparison table
    print("\n" + "=" * 80)
    print("TEMPLATE COMPARISON RESULTS")
    print("=" * 80)
    metric_keys = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    header = f"{'Template':<20}" + "".join(f"{k:>12}" for k in metric_keys)
    print(header)
    print("-" * 80)
    for tid, data in comparison["templates"].items():
        m = data["metrics"]
        row = f"{tid:<20}" + "".join(f"{m.get(k, 0.0):>12.4f}" for k in metric_keys)
        print(row)
    print("=" * 80)
    print()

    # Save aggregated results
    out_file = results_dir / f"template_comparison_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison saved to %s", out_file)


if __name__ == "__main__":
    main()
