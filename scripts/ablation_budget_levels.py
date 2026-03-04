#!/usr/bin/env python
"""Ablation study: sweep budget as a fraction of total queries.

Usage::

    python -m scripts.ablation_budget_levels --fractions 0,0.25,0.5,1.0 --num-queries 10
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
from src.rl.budget_experiment import BudgetExperiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_sweep(
    fractions: list[float],
    num_queries: int,
    num_candidates: int | None = None,
    random_seed: int | None = None,
) -> list[dict]:
    """Run budget fraction sweep and return results for each fraction."""
    results = []
    for fraction in fractions:
        budget = max(0, round(fraction * num_queries))
        logger.info(
            f"Running BudgetExperiment with fraction={fraction:.0%}, budget={budget}"
        )
        experiment = BudgetExperiment(
            total_budget=budget,
            num_test_queries=num_queries,
            num_candidates=num_candidates,
            random_seed=random_seed,
        )
        result = experiment.run()

        emb_metrics = result["embedding_metrics"]
        rl_metrics = result["rl_budget_metrics"]
        summary = result["budget_agent_summary"]

        results.append(
            {
                "fraction": fraction,
                "budget": budget,
                "metrics": rl_metrics,
                "embedding_metrics": emb_metrics,
                "llm_calls": summary.get("llm_queries", 0),
                "llm_fraction": summary.get("llm_fraction", 0.0),
            }
        )
        logger.info(
            f"fraction={fraction:.0%}: MRR={rl_metrics.get('MRR', 0):.4f}, "
            f"LLM calls={summary.get('llm_queries', 0)}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study over budget levels (as fraction of queries)"
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0,0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated list of budget fractions (0–1)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of test queries to evaluate",
    )
    args = parser.parse_args()

    fractions = [float(f.strip()) for f in args.fractions.split(",")]
    settings = get_settings()

    logger.info(f"Ablation sweep over budget fractions: {fractions}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = run_sweep(
        fractions=fractions,
        num_queries=args.num_queries,
        random_seed=settings.random_seed,
    )

    # Print comparison table
    print("\n" + "=" * 75)
    print("ABLATION: BUDGET LEVELS")
    print("=" * 75)
    print(
        f"{'Fraction':>10} {'Budget':>8} {'Emb MRR':>10} "
        f"{'RL MRR':>10} {'LLM Calls':>10} {'LLM %':>8}"
    )
    print("-" * 75)
    for row in results:
        em = row["embedding_metrics"]
        rm = row["metrics"]
        print(
            f"{row['fraction']:>10.0%} {row['budget']:>8} "
            f"{em.get('MRR', 0):>10.4f} {rm.get('MRR', 0):>10.4f} "
            f"{row['llm_calls']:>10} {row['llm_fraction']:>8.1%}"
        )
    print("=" * 75)

    # Save results
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"ablation_budget_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": {
            "fractions": fractions,
            "num_queries": args.num_queries,
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
