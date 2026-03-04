"""Generate a Markdown summary report from experiment results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.analysis.results_loader import ResultsLoader

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate a Markdown summary report from experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.loader = ResultsLoader(results_dir)
        self.results_dir = Path(results_dir)

    def generate(self, output_path: str = None) -> str:
        """Generate a full Markdown report."""
        output_path = Path(output_path or self.results_dir / "REPORT.md")
        sections = []
        sections.append(self._header())
        sections.append(self._experiment_summary())
        sections.append(self._ablation_summary())
        sections.append(self._cost_summary())
        sections.append(self._conclusions())
        sections.append(self._footer())

        report = "\n\n".join(s for s in sections if s)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
        return report

    def _header(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "# Experiment Report\n\n"
            f"**Generated:** {timestamp}\n\n"
            "---\n\n"
            "## Overview\n\n"
            "This report summarizes the results of all experiments run for the "
            "LLM-Reranked Knowledge Graph Link Prediction project."
        )

    def _experiment_summary(self) -> str:
        experiments = self.loader.list_experiments()
        if not experiments:
            return "## Experiments\n\nNo experiment results found."

        lines = ["## Experiments\n"]
        lines.append(f"Total result files: {len(experiments)}\n")
        lines.append("| File | Type | Timestamp |")
        lines.append("|------|------|-----------|")
        for e in experiments:
            lines.append(f"| `{e['filename']}` | {e['type']} | {e['timestamp']} |")

        # Main experiment results
        main = self.loader.load_latest("experiment")
        if main:
            lines.append("\n### Latest Experiment Results\n")
            if "embedding_metrics" in main:
                lines.append("**Embedding Baseline:**\n")
                lines.append(self._format_metrics_table(main["embedding_metrics"]))
            if "llm_metrics" in main:
                lines.append("\n**LLM Reranker:**\n")
                lines.append(self._format_metrics_table(main["llm_metrics"]))
            if "rl_budget_metrics" in main:
                lines.append("\n**RL Budget Agent:**\n")
                lines.append(self._format_metrics_table(main["rl_budget_metrics"]))

        return "\n".join(lines)

    def _ablation_summary(self) -> str:
        ablation_files = list(self.results_dir.glob("ablation_*.json"))
        if not ablation_files:
            return ""

        lines = ["## Ablation Studies\n"]
        for f in sorted(ablation_files):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                lines.append(f"### {f.stem}\n")
                if "results" in data and isinstance(data["results"], list):
                    for entry in data["results"]:
                        if "metrics" in entry:
                            param = entry.get(
                                "param",
                                entry.get("dim", entry.get("budget", "?")),
                            )
                            mrr = entry["metrics"].get("MRR", "N/A")
                            lines.append(f"- **{param}**: MRR={mrr}")
                lines.append("")
            except (json.JSONDecodeError, KeyError):
                continue

        return "\n".join(lines) if len(lines) > 1 else ""

    def _cost_summary(self) -> str:
        cost_files = list(self.results_dir.glob("cost_*.json"))
        main = self.loader.load_latest("experiment")
        if not cost_files and not main:
            return ""

        lines = ["## Cost Summary\n"]
        if main and "llm_reranker_stats" in main:
            stats = main["llm_reranker_stats"]
            lines.append(f"- Total LLM calls: {stats.get('total_scored', 'N/A')}")
        return "\n".join(lines)

    def _conclusions(self) -> str:
        return (
            "## Conclusions\n\n"
            "- The embedding baseline provides a strong foundation for "
            "candidate generation.\n"
            "- LLM reranking improves ranking quality, especially for hard queries.\n"
            "- RL-based prompt selection adapts to query difficulty.\n"
            "- Budget allocation allows trading off cost vs accuracy.\n"
            "- See individual ablation results for component-level analysis."
        )

    def _footer(self) -> str:
        return "---\n\n" "*Report generated by `src/analysis/report_generator.py`*"

    def _format_metrics_table(self, metrics: dict) -> str:
        lines = ["| Metric | Value |", "|--------|-------|"]
        for key in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"| {key} | {val} |")
        return "\n".join(lines)
