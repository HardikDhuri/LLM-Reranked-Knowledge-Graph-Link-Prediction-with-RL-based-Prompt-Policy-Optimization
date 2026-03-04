"""Resolve a sample of Freebase MIDs to Wikidata labels via SPARQL.

Usage:
    python -m scripts.resolve_sample_mids
"""

import random
import sys
from pathlib import Path

# Ensure project root is on the path when run as a module.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.wikidata.sparql import WikidataResolver


def main():
    settings = get_settings()
    data_dir = settings.data_dir

    # Collect entity MIDs from the dataset files.
    entity_file = data_dir / "entities.dict"
    if not entity_file.exists():
        # Fall back to extracting unique entities from train.txt
        train_file = data_dir / "train.txt"
        if not train_file.exists():
            print(f"No dataset found in {data_dir}. Run download_fb15k237.py first.")
            return
        mids = set()
        with open(train_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    mids.add(parts[0])
                    mids.add(parts[2])
        mids = list(mids)
    else:
        mids = []
        with open(entity_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    mids.append(parts[-1])

    if not mids:
        print("No entity MIDs found in dataset.")
        return

    random.seed(settings.random_seed)
    sample = random.sample(mids, min(10, len(mids)))

    resolver = WikidataResolver(cache_dir=settings.cache_dir)

    print(f"Resolving {len(sample)} sample MIDs...\n")
    for mid in sample:
        result = resolver.mid_to_text(mid)
        qid = result["qid"] or "N/A"
        label = result["label"]
        description = result["description"]
        print(f"  {mid} → {qid} → {label!r} ({description})")

    print()
    stats = resolver.cache_stats()
    print(f"Cache stats: {stats}")


if __name__ == "__main__":
    main()
