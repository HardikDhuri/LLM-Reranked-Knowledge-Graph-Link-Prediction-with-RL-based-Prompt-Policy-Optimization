#!/usr/bin/env python
"""Ablation study: sweep LinUCB alpha (exploration parameter) for RLPromptSelector.

Usage::

    python -m scripts.ablation_alpha --alphas 0.1,1.0,5.0 --num-queries 15
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
from src.prompts.renderer import PromptManager
from src.rl.features import QueryFeatureExtractor
from src.rl.prompt_selector import RLPromptSelector
from src.wikidata.sparql import WikidataResolver

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_sweep(
    alphas: list[float],
    num_queries: int,
    num_candidates: int,
    random_seed: int | None = None,
) -> list[dict]:
    """Run alpha sweep and return results for each alpha value."""
    import random

    dataset = load_fb15k237()
    all_entities = dataset.entities
    known_triples = dataset.all_triples_set

    logger.info("Training embedding baseline...")
    embedding = EmbeddingBaseline(
        train_triples=dataset.train,
        valid_triples=dataset.valid,
        num_epochs=50,
        embedding_dim=128,
        random_seed=random_seed,
    )
    embedding.train()

    random.seed(random_seed)
    test_queries = random.sample(dataset.test, min(num_queries, len(dataset.test)))

    feature_extractor = QueryFeatureExtractor.from_triples(dataset.train)
    resolver = WikidataResolver()
    llm_client = LLMClient()
    prompt_manager = PromptManager()

    results = []
    for alpha in alphas:
        logger.info(f"Evaluating with alpha={alpha}")
        random.seed(random_seed)

        selector = RLPromptSelector(
            llm_client=llm_client,
            resolver=resolver,
            prompt_manager=prompt_manager,
            feature_extractor=feature_extractor,
            agent_type="linucb",
            alpha=alpha,
        )

        ranking_results = []
        cumulative_reward = 0.0

        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, num_candidates, seed=random_seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)

            template_id, scored, true_rank = selector.select_and_score(
                h, r, t, candidates
            )
            cumulative_reward += 1.0 / true_rank
            ranking_results.append(
                RankingResult(
                    query=(h, r, t),
                    true_rank=true_rank,
                    num_candidates=len(candidates),
                    scored_candidates=scored,
                )
            )

        metrics = compute_all_metrics(ranking_results)
        summary = selector.summary()
        arm_dist = summary["agent"].get("arm_selection_counts", {})

        results.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "cumulative_reward": round(cumulative_reward, 4),
                "arm_distribution": arm_dist,
            }
        )
        logger.info(
            f"alpha={alpha}: MRR={metrics.get('MRR', 0):.4f}, "
            f"reward={cumulative_reward:.4f}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study over LinUCB alpha (exploration parameter)"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.1,0.5,1.0,2.0",
        help="Comma-separated list of alpha values",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of test queries to evaluate",
    )
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    settings = get_settings()
    num_candidates = settings.num_candidates

    logger.info(f"Ablation sweep over alphas: {alphas}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = run_sweep(
        alphas=alphas,
        num_queries=args.num_queries,
        num_candidates=num_candidates,
        random_seed=settings.random_seed,
    )

    # Print comparison table
    print("\n" + "=" * 65)
    print("ABLATION: LINUCB ALPHA")
    print("=" * 65)
    print(
        f"{'Alpha':>8} {'MRR':>10} {'Hits@1':>10} "
        f"{'Hits@10':>10} {'Cum Reward':>12}"
    )
    print("-" * 65)
    for row in results:
        m = row["metrics"]
        print(
            f"{row['alpha']:>8.3f} {m.get('MRR', 0):>10.4f} "
            f"{m.get('Hits@1', 0):>10.4f} {m.get('Hits@10', 0):>10.4f} "
            f"{row['cumulative_reward']:>12.4f}"
        )
    print("=" * 65)

    # Save results
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"ablation_alpha_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": {
            "alphas": alphas,
            "num_queries": args.num_queries,
            "num_candidates": num_candidates,
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
