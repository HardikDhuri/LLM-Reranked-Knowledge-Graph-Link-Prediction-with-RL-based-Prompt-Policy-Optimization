#!/usr/bin/env python
"""Load the FB15k-237 dataset and print a summary with sample triples.

Usage::

    python -m scripts.smoke_load_dataset
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_fb15k237  # noqa: E402


def main() -> None:
    print("Loading FB15k-237 dataset …")
    dataset = load_fb15k237()

    summary = dataset.summary()
    print("\nDataset summary:")
    print(
        f"  Train: {summary['train']}, Valid: {summary['valid']}, "
        f"Test: {summary['test']}, Entities: {summary['entities']}, "
        f"Relations: {summary['relations']}"
    )
    print(f"  All triples: {summary['all']}")

    for split_name, triples in [
        ("train", dataset.train),
        ("valid", dataset.valid),
        ("test", dataset.test),
    ]:
        print(f"\nSample triples from {split_name}:")
        for triple in triples[:5]:
            print(f"  {triple}")


if __name__ == "__main__":
    main()
