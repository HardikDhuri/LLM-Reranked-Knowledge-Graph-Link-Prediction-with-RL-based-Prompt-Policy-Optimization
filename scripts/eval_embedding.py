#!/usr/bin/env python
"""Evaluate the RotatE embedding baseline on FB15k-237.

Trains a quick embedding model, then runs the evaluation harness on a
sample of test queries and prints MRR + Hits@K.

Usage::

    python -m scripts.eval_embedding
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.data.fb15k237 import load_fb15k237
from src.eval.evaluate import rank_tail_query
from src.eval.metrics import compute_all_metrics, format_metrics
from src.models.embedding_baseline import EmbeddingBaseline


def main() -> None:
    settings = get_settings()

    print("Loading FB15k-237 dataset…")
    dataset = load_fb15k237()
    print(f"Dataset loaded: {dataset.summary()}")

    print("\nTraining embedding baseline (epochs=50, dim=128)…")
    baseline = EmbeddingBaseline(
        train_triples=dataset.train,
        valid_triples=dataset.valid,
        embedding_dim=128,
        num_epochs=50,
        batch_size=256,
    )
    summary = baseline.train()
    print("Training summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    score_fn = baseline.get_score_fn()
    all_entities = dataset.entities
    known_triples = dataset.all_triples_set

    # Sample test queries
    n = settings.sample_test_queries
    rng = random.Random(settings.random_seed)
    test_sample = rng.sample(dataset.test, min(n, len(dataset.test)))

    print(f"\nEvaluating on {len(test_sample)} test queries…")
    results = []
    for query in test_sample:
        result = rank_tail_query(
            query=query,
            score_fn=score_fn,
            all_entities=all_entities,
            known_triples=known_triples,
            num_candidates=settings.num_candidates,
            seed=settings.random_seed,
        )
        results.append(result)

    metrics = compute_all_metrics(results)
    print("\nEmbedding Baseline Metrics:")
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
