#!/usr/bin/env python
"""Demonstrate evaluation on a tiny synthetic knowledge graph.

Usage::

    python -m scripts.eval_dummy
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import RankingResult, compute_all_metrics, format_metrics, rank_tail_query

# ---------------------------------------------------------------------------
# Build a tiny synthetic KG
# ---------------------------------------------------------------------------

ENTITIES = [f"e{i}" for i in range(10)]
RELATIONS = ["r0", "r1", "r2"]

rng = random.Random(42)
ALL_TRIPLES = list(
    {
        (rng.choice(ENTITIES), rng.choice(RELATIONS), rng.choice(ENTITIES))
        for _ in range(20)
    }
)

# Pick 5 test triples (they are already in ALL_TRIPLES, so filtered eval applies)
TEST_TRIPLES = ALL_TRIPLES[:5]
KNOWN_TRIPLES = set(ALL_TRIPLES)


def dummy_score(h: str, r: str, t: str) -> float:
    """Random scorer — just for demonstration."""
    return random.random()


def main() -> None:
    results: list[RankingResult] = []
    for triple in TEST_TRIPLES:
        result = rank_tail_query(
            query=triple,
            score_fn=dummy_score,
            all_entities=ENTITIES,
            known_triples=KNOWN_TRIPLES,
            num_candidates=len(ENTITIES),
            seed=0,
        )
        results.append(result)
        print(
            f"Query {triple}: true_rank={result.true_rank}/{result.num_candidates}"
        )

    metrics = compute_all_metrics(results)
    print("\nMetrics:")
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
