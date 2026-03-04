"""Load and aggregate experiment results from the results directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsLoader:
    """Load and aggregate experiment results from the results directory."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def list_experiments(self) -> list[dict]:
        """List all experiment result files with metadata."""
        experiments = []
        for f in sorted(self.results_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                experiments.append(
                    {
                        "filename": f.name,
                        "path": str(f),
                        "timestamp": data.get("timestamp", "unknown"),
                        "type": self._infer_type(f.name),
                        "config": data.get("config", {}),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return experiments

    def load_experiment(self, filename: str) -> dict:
        """Load a single experiment result."""
        filepath = self.results_dir / filename
        with open(filepath) as f:
            return json.load(f)

    def load_latest(self, experiment_type: str = None) -> dict | None:
        """Load the most recent experiment of a given type."""
        experiments = self.list_experiments()
        if experiment_type:
            experiments = [e for e in experiments if e["type"] == experiment_type]
        if not experiments:
            return None
        latest = max(experiments, key=lambda e: e["timestamp"])
        return self.load_experiment(latest["filename"])

    def load_ablation(self, ablation_name: str) -> list[dict]:
        """Load all results for a specific ablation study."""
        results = []
        for f in sorted(self.results_dir.glob(f"ablation_{ablation_name}_*.json")):
            with open(f) as fh:
                results.append(json.load(fh))
        return results

    def _infer_type(self, filename: str) -> str:
        if "ablation" in filename:
            return "ablation"
        elif "budget" in filename:
            return "budget"
        elif "experiment" in filename:
            return "experiment"
        elif "cost" in filename:
            return "cost"
        else:
            return "other"
