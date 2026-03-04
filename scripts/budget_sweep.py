#!/usr/bin/env python
"""Sweep across multiple budget levels and report MRR at each level.

Usage::

    python -m scripts.budget_sweep --num-queries 20 --num-candidates 10
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep across budget levels and compare MRR"
    )
    parser.add_argument(
        "--num-queries", type=int, default=None, help="Number of test queries"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of candidates per query",
    )
    parser.add_argument(
        "--agent-type",
        choices=["linucb", "epsilon_greedy"],
        default="linucb",
        help="RL agent type",
    )
    parser.add_argument(
        "--template", type=str, default="minimal", help="Prompt template ID"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="LinUCB exploration parameter"
    )
    args = parser.parse_args()

    settings = get_settings()
    num_queries = args.num_queries or settings.sample_test_queries

    # Budget levels as fractions of total queries (0%, 5%, 10%, 25%, 50%, 100%)
    budget_fractions = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    budget_levels = sorted(
        {max(0, round(f * num_queries)) for f in budget_fractions}
    )

    logger.info(f"Budget sweep over levels: {budget_levels}")
    logger.info(f"Num queries: {num_queries}")

    sweep_results = []

    for budget in budget_levels:
        logger.info(f"\n--- Running with budget={budget} ---")
        experiment = BudgetExperiment(
            template_id=args.template,
            total_budget=budget,
            agent_type=args.agent_type,
            alpha=args.alpha,
            num_test_queries=num_queries,
            num_candidates=args.num_candidates,
        )
        result = experiment.run()
        sweep_results.append(
            {
                "budget": budget,
                "budget_fraction": budget / max(1, num_queries),
                "embedding_mrr": result["embedding_metrics"].get("MRR", 0),
                "rl_budget_mrr": result["rl_budget_metrics"].get("MRR", 0),
                "llm_queries": result["budget_agent_summary"]["llm_queries"],
                "llm_fraction": result["budget_agent_summary"]["llm_fraction"],
            }
        )

    # Print sweep table
    print("\n" + "=" * 75)
    print("BUDGET SWEEP RESULTS")
    print("=" * 75)
    print(
        f"{'Budget':>8} {'Fraction':>10} {'Emb MRR':>12} {'RL MRR':>12} "
        f"{'LLM Calls':>10} {'LLM %':>8}"
    )
    print("-" * 75)
    for row in sweep_results:
        print(
            f"{row['budget']:>8} {row['budget_fraction']:>10.1%} "
            f"{row['embedding_mrr']:>12.4f} {row['rl_budget_mrr']:>12.4f} "
            f"{row['llm_queries']:>10} {row['llm_fraction']:>8.1%}"
        )
    print("=" * 75)

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = results_dir / f"budget_sweep_{timestamp}.json"
    with open(sweep_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    "num_queries": num_queries,
                    "num_candidates": args.num_candidates,
                    "agent_type": args.agent_type,
                    "template": args.template,
                    "budget_levels": budget_levels,
                },
                "sweep_results": sweep_results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Sweep results saved to {sweep_file}")


if __name__ == "__main__":
    main()
