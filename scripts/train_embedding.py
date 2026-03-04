#!/usr/bin/env python
"""Train a RotatE embedding baseline on FB15k-237.

Usage::

    python -m scripts.train_embedding --epochs 50 --dim 128
    python -m scripts.train_embedding --epochs 5 --dim 64
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fb15k237 import load_fb15k237
from src.models.embedding_baseline import EmbeddingBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RotatE embedding baseline")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading FB15k-237 dataset…")
    dataset = load_fb15k237()
    print(f"Dataset loaded: {dataset.summary()}")

    print(f"\nInitialising EmbeddingBaseline (dim={args.dim}, epochs={args.epochs})…")
    baseline = EmbeddingBaseline(
        train_triples=dataset.train,
        valid_triples=dataset.valid,
        embedding_dim=args.dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\nTraining…")
    summary = baseline.train()
    print("\nTraining summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Score 5 random test triples
    rng = random.Random(42)
    sample = rng.sample(dataset.test, min(5, len(dataset.test)))
    score_fn = baseline.get_score_fn()

    print("\nSample test-triple scores:")
    for h, r, t in sample:
        score = score_fn(h, r, t)
        print(f"  ({h}, {r}, {t})  →  {score:.4f}")

    # Save model
    baseline.save_model()
    print("\nModel saved.")


if __name__ == "__main__":
    main()
