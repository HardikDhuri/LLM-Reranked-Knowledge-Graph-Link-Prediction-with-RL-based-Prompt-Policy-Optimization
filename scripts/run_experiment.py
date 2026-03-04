#!/usr/bin/env python
"""CLI for running the full KG link prediction experiment pipeline.

Usage::

    python -m scripts.run_experiment --template minimal \\
        --num-queries 5 --num-candidates 10
    python -m scripts.run_experiment --list-templates
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import ExperimentRunner
from src.prompts.renderer import PromptManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full KG link prediction experiment pipeline."
    )
    parser.add_argument(
        "--template",
        default="minimal",
        help="Prompt template ID to use for LLM reranking (default: minimal)",
    )
    parser.add_argument(
        "--embedding-epochs",
        type=int,
        default=50,
        help="Number of training epochs for the embedding baseline (default: 50)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for the baseline model (default: 128)",
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
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List all available prompt templates and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_templates:
        registry = PromptManager()
        print("Available prompt templates:")
        for tid in registry.list_ids():
            tmpl = registry.get(tid)
            print(f"  {tid}: {tmpl.name} — {tmpl.description}")
        return

    runner = ExperimentRunner(
        template_id=args.template,
        embedding_epochs=args.embedding_epochs,
        embedding_dim=args.embedding_dim,
        num_test_queries=args.num_queries,
        num_candidates=args.num_candidates,
        random_seed=args.seed,
    )
    runner.run()


if __name__ == "__main__":
    main()
