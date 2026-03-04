#!/usr/bin/env python
"""Generate a Markdown summary report from all experiment results.

Usage::

    python -m scripts.generate_report
    python -m scripts.generate_report --results-dir results --output results/REPORT.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary report from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the Markdown report (default: <results-dir>/REPORT.md)",
    )
    args = parser.parse_args()

    generator = ReportGenerator(results_dir=args.results_dir)
    report = generator.generate(output_path=args.output)
    print(report)


if __name__ == "__main__":
    main()
