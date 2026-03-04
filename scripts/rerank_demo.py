#!/usr/bin/env python
"""Demonstrate LLM reranking on 5 FB15k-237 test triples.

Usage::

    python -m scripts.rerank_demo
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fb15k237 import load_fb15k237
from src.eval.candidates import generate_tail_candidates
from src.eval.metrics import compute_all_metrics, format_metrics
from src.models import LLMReranker, TripleScorer
from src.models.llm_client import LLMClient
from src.prompts.renderer import TemplateRegistry
from src.wikidata.sparql import WikidataResolver

NUM_QUERIES = 5
NUM_CANDIDATES = 10
TEMPLATE_ID = "minimal"
SEED = 42


def main() -> None:
    print("Loading FB15k-237 dataset …")
    dataset = load_fb15k237()
    print(f"Loaded {len(dataset.test)} test triples.")

    # Sample test triples
    rng = random.Random(SEED)
    test_triples = rng.sample(dataset.test, NUM_QUERIES)

    all_entities = dataset.entities

    # Build components
    llm = LLMClient()
    resolver = WikidataResolver()
    registry = TemplateRegistry()
    template = registry.get(TEMPLATE_ID)

    scorer = TripleScorer(llm_client=llm, resolver=resolver, template=template)
    reranker = LLMReranker(scorer=scorer)

    # Build candidate sets
    candidates_per_query: dict = {}
    for h, r, t in test_triples:
        query = (h, r, t)
        cands = generate_tail_candidates(
            h=h,
            r=r,
            true_t=t,
            all_entities=all_entities,
            num_candidates=NUM_CANDIDATES,
            seed=SEED,
        )
        candidates_per_query[query] = cands

    # Rerank
    print(
        f"\nReranking {NUM_QUERIES} queries with {NUM_CANDIDATES} candidates each …\n"
    )
    results = reranker.rerank_batch(
        queries=test_triples,
        candidates_per_query=candidates_per_query,
        progress=True,
    )

    # Per-query output
    for result in results:
        h, r, t = result.query
        print(f"Query: ({h}, {r}, ?)")
        print(f"  True tail: {t}")
        print(f"  True rank: {result.true_rank} / {result.num_candidates}")
        print("  Top-3 scored candidates:")
        for entity, score in result.scored_candidates[:3]:
            marker = " <-- TRUE" if entity == t else ""
            print(f"    {entity}: {score:.4f}{marker}")
        print()

    # Aggregate metrics
    metrics = compute_all_metrics(results)
    print("Aggregate metrics:")
    print(format_metrics(metrics))

    print("\nReranker stats:")
    for k, v in reranker.stats().items():
        if k != "scorer_stats":
            print(f"  {k}: {v}")
    print("Scorer stats:")
    for k, v in scorer.stats().items():
        if k != "llm_stats":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
