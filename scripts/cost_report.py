"""Aggregate and print LLM cost summaries from saved cost JSON files."""

import argparse
import json
import sys
from pathlib import Path


def find_cost_files(results_dir: Path) -> list[Path]:
    """Find all cost JSON files under results_dir."""
    return sorted(results_dir.rglob("cost_*.json"))


def load_cost_file(filepath: Path) -> dict:
    with open(filepath) as f:
        return json.load(f)


def aggregate_costs(cost_data: list[dict]) -> dict:
    """Aggregate cost summaries across multiple experiments."""
    totals: dict = {
        "experiments": len(cost_data),
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,
        "total_latency_s": 0.0,
    }
    for d in cost_data:
        s = d.get("summary", {})
        totals["total_calls"] += s.get("total_calls", 0)
        totals["successful_calls"] += s.get("successful_calls", 0)
        totals["failed_calls"] += s.get("failed_calls", 0)
        totals["total_tokens"] += s.get("total_tokens", 0)
        totals["total_prompt_tokens"] += s.get("total_prompt_tokens", 0)
        totals["total_completion_tokens"] += s.get("total_completion_tokens", 0)
        totals["total_cost_usd"] += s.get("total_cost_usd", 0.0)
        totals["total_latency_s"] += s.get("total_latency_s", 0.0)
    totals["total_cost_usd"] = round(totals["total_cost_usd"], 6)
    totals["total_latency_s"] = round(totals["total_latency_s"], 2)
    return totals


def print_report(files: list[Path], cost_data: list[dict], totals: dict) -> None:
    print("\n" + "=" * 60)
    print("LLM COST REPORT")
    print("=" * 60)
    print(f"Results directory scanned: {len(files)} cost file(s) found\n")

    print("PER-EXPERIMENT BREAKDOWN:")
    print("-" * 60)
    for filepath, d in zip(files, cost_data):
        s = d.get("summary", {})
        print(f"  File: {filepath.name}")
        print(f"    Model:         {s.get('model', 'unknown')}")
        print(f"    Calls:         {s.get('total_calls', 0)}")
        print(f"    Tokens:        {s.get('total_tokens', 0):,}")
        print(f"    Cost:          ${s.get('total_cost_usd', 0.0):.4f}")
        print(f"    Latency:       {s.get('total_latency_s', 0.0):.1f}s")
        print()

    print("AGGREGATE TOTALS:")
    print("-" * 60)
    print(f"  Experiments:       {totals['experiments']}")
    print(
        f"  Total calls:       {totals['total_calls']} "
        f"({totals['successful_calls']} ok, {totals['failed_calls']} failed)"
    )
    print(
        f"  Total tokens:      {totals['total_tokens']:,} "
        f"(prompt: {totals['total_prompt_tokens']:,}, "
        f"completion: {totals['total_completion_tokens']:,})"
    )
    print(f"  Total cost:        ${totals['total_cost_usd']:.4f}")
    print(f"  Total latency:     {totals['total_latency_s']:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and print LLM cost summaries."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to scan for cost JSON files (default: from config)",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        try:
            from src.config import get_settings

            results_dir = get_settings().results_dir
        except Exception:
            results_dir = Path("results")

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_cost_files(results_dir)
    if not files:
        print(f"No cost JSON files found in {results_dir}")
        return

    cost_data = [load_cost_file(f) for f in files]
    totals = aggregate_costs(cost_data)
    print_report(files, cost_data, totals)


if __name__ == "__main__":
    main()
