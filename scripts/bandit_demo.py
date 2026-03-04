#!/usr/bin/env python
"""Demonstrate RL-based prompt selection with a LinUCB bandit agent.

Usage::

    python -m scripts.bandit_demo --num-queries 20 --num-candidates 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fb15k237 import load_fb15k237
from src.models.llm_client import LLMClient
from src.prompts.renderer import PromptManager
from src.rl.features import QueryFeatureExtractor
from src.rl.prompt_selector import RLPromptSelector
from src.wikidata.sparql import WikidataResolver

SEED = 42


def generate_candidates(
    h: str,
    r: str,
    true_t: str,
    all_entities: list[str],
    num_candidates: int,
    rng: random.Random,
) -> list[str]:
    """Sample random candidates ensuring the true tail is included."""
    pool = [e for e in all_entities if e != true_t]
    sampled = rng.sample(pool, min(num_candidates - 1, len(pool)))
    candidates = sampled + [true_t]
    rng.shuffle(candidates)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit prompt selection demo")
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument(
        "--agent-type",
        choices=["linucb", "epsilon_greedy"],
        default="linucb",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    print("Loading FB15k-237 dataset …")
    dataset = load_fb15k237()
    print(f"  Train triples: {len(dataset.train)}")
    print(f"  Test  triples: {len(dataset.test)}")

    # Build feature extractor from training triples
    print("Building feature extractor from training triples …")
    feature_extractor = QueryFeatureExtractor.from_triples(dataset.train)
    print(f"  Entities: {feature_extractor.num_entities}")
    print(f"  Relations: {feature_extractor.num_relations}")
    print(f"  Feature dim: {feature_extractor.feature_dim}")

    # Sample test queries
    rng = random.Random(SEED)
    test_triples = rng.sample(dataset.test, min(args.num_queries, len(dataset.test)))
    all_entities = dataset.entities

    # Build components
    llm = LLMClient()
    resolver = WikidataResolver()
    prompt_manager = PromptManager()

    print(f"\nTemplate IDs: {prompt_manager.list_ids()}")

    selector = RLPromptSelector(
        llm_client=llm,
        resolver=resolver,
        prompt_manager=prompt_manager,
        feature_extractor=feature_extractor,
        agent_type=args.agent_type,
        alpha=args.alpha,
        epsilon=args.epsilon,
    )

    print(
        f"\nRunning {args.num_queries} queries with {args.num_candidates} candidates "
        f"each (agent: {args.agent_type}) …\n"
    )

    cumulative_reward = 0.0
    cumulative_rewards: list[float] = []

    for step, (h, r, t) in enumerate(test_triples, start=1):
        candidates = generate_candidates(
            h, r, t, all_entities, args.num_candidates, rng
        )
        template_id, scored, true_rank = selector.select_and_score(
            h=h,
            r=r,
            true_t=t,
            candidates=candidates,
        )
        reward = 1.0 / true_rank
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward / step)

        print(
            f"Step {step:3d} | template={template_id:<20s} | "
            f"rank={true_rank:3d}/{len(candidates)} | "
            f"reward={reward:.4f} | avg_reward={cumulative_rewards[-1]:.4f}"
        )

    print("\n--- Agent Summary ---")
    summary = selector.summary()
    agent_summary = summary["agent"]
    print(f"Total steps : {agent_summary['total_steps']}")
    if "alpha" in agent_summary:
        print(f"Alpha       : {agent_summary['alpha']}")
    if "epsilon" in agent_summary:
        print(f"Epsilon     : {agent_summary['epsilon']}")

    print("\nArm statistics:")
    col_names = (
        f"{'Arm':<4} {'Name':<25} {'Count':>6} {'Total Reward':>14} {'Avg Reward':>12}"
    )
    header = "  " + col_names
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stat in agent_summary["arm_stats"]:
        print(
            f"  {stat['arm']:<4} {stat['name']:<25} {stat['count']:>6} "
            f"{stat['total_reward']:>14.4f} {stat['avg_reward']:>12.4f}"
        )

    if cumulative_rewards:
        print(f"\nFinal cumulative average reward: {cumulative_rewards[-1]:.4f}")
    else:
        print("\nNo queries were processed.")


if __name__ == "__main__":
    main()
