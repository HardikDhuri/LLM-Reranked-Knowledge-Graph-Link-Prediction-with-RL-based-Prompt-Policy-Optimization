#!/usr/bin/env python
"""Ablation study: sweep embedding dimensions and measure link prediction quality.

Usage::

    python -m scripts.ablation_embedding_dim \\
        --dims 32,64,128 --epochs 20 --num-queries 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.data.fb15k237 import load_fb15k237
from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import RankingResult, compute_all_metrics
from src.models.embedding_baseline import EmbeddingBaseline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_sweep(
    dims: list[int],
    epochs: int,
    num_queries: int,
    num_candidates: int,
    random_seed: int | None = None,
) -> list[dict]:
    """Run embedding dim sweep and return results for each dim."""
    import random

    dataset = load_fb15k237()
    all_entities = dataset.entities
    known_triples = dataset.all_triples_set

    random.seed(random_seed)
    test_queries = random.sample(dataset.test, min(num_queries, len(dataset.test)))

    results = []
    for dim in dims:
        logger.info(f"Training embedding with dim={dim}, epochs={epochs}")
        start = time.time()
        embedding = EmbeddingBaseline(
            train_triples=dataset.train,
            valid_triples=dataset.valid,
            num_epochs=epochs,
            embedding_dim=dim,
            random_seed=random_seed,
        )
        embedding.train()
        train_time = time.time() - start

        score_fn = embedding.get_score_fn()
        ranking_results = []
        for h, r, t in test_queries:
            candidates = generate_tail_candidates(
                h, r, t, all_entities, num_candidates, seed=random_seed
            )
            candidates = filter_candidates_tail(h, r, candidates, t, known_triples)
            scored = [(tail, score_fn(h, r, tail)) for tail in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [tail for tail, _ in scored]
            rank = ranked.index(t) + 1 if t in ranked else len(ranked) + 1
            ranking_results.append(
                RankingResult(
                    query=(h, r, t),
                    true_rank=rank,
                    num_candidates=len(candidates),
                    scored_candidates=scored,
                )
            )

        metrics = compute_all_metrics(ranking_results)
        results.append(
            {
                "dim": dim,
                "metrics": metrics,
                "train_time_s": round(train_time, 2),
            }
        )
        logger.info(
            f"dim={dim}: MRR={metrics.get('MRR', 0):.4f}, "
            f"Hits@10={metrics.get('Hits@10', 0):.4f}, "
            f"time={train_time:.1f}s"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study over embedding dimensions"
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="32,64,128,256",
        help="Comma-separated list of embedding dimensions",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs per dimension"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=15,
        help="Number of test queries to evaluate",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=25,
        help="Number of candidates per query",
    )
    args = parser.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",")]
    settings = get_settings()
    num_candidates = args.num_candidates or settings.num_candidates

    logger.info(f"Ablation sweep over dims: {dims}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = run_sweep(
        dims=dims,
        epochs=args.epochs,
        num_queries=args.num_queries,
        num_candidates=num_candidates,
        random_seed=settings.random_seed,
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION: EMBEDDING DIMENSION")
    print("=" * 70)
    print(
        f"{'Dim':>6} {'MRR':>10} {'Hits@1':>10} {'Hits@3':>10} "
        f"{'Hits@10':>10} {'Time(s)':>10}"
    )
    print("-" * 70)
    for row in results:
        m = row["metrics"]
        print(
            f"{row['dim']:>6} {m.get('MRR', 0):>10.4f} {m.get('Hits@1', 0):>10.4f} "
            f"{m.get('Hits@3', 0):>10.4f} {m.get('Hits@10', 0):>10.4f} "
            f"{row['train_time_s']:>10.1f}"
        )
    print("=" * 70)

    # Save results
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"ablation_embedding_dim_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": {
            "dims": dims,
            "epochs": args.epochs,
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
