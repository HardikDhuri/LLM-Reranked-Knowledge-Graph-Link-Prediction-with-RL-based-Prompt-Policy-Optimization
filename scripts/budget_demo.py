#!/usr/bin/env python
"""Run a budget-constrained RL experiment comparing embedding-only vs RL agent.

Usage::

    python -m scripts.budget_demo --budget 10 --num-queries 20 --num-candidates 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.budget_experiment import BudgetExperiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budget-constrained RL vs embedding-only experiment"
    )
    parser.add_argument(
        "--budget", type=int, default=50, help="Total LLM call budget"
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
        "--num-queries", type=int, default=None, help="Number of test queries"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of candidates per query",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="LinUCB exploration parameter"
    )
    args = parser.parse_args()

    experiment = BudgetExperiment(
        template_id=args.template,
        total_budget=args.budget,
        agent_type=args.agent_type,
        alpha=args.alpha,
        num_test_queries=args.num_queries,
        num_candidates=args.num_candidates,
    )
    experiment.run()


if __name__ == "__main__":
    main()
