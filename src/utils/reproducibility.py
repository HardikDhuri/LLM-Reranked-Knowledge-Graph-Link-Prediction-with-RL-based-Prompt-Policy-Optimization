"""Reproducibility utilities for experiment management."""

import hashlib
import json
import logging
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"All random seeds set to {seed}")


def get_environment_info() -> dict:
    """Capture environment info for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
    try:
        import pykeen

        info["pykeen_version"] = pykeen.get_version()
    except (ImportError, AttributeError):
        info["pykeen_version"] = "not installed"
    return info


def compute_config_hash(config: dict) -> str:
    """Compute a deterministic hash of an experiment config."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def save_experiment_manifest(
    config: dict,
    results: dict,
    output_dir: Path,
    experiment_name: str = None,
) -> Path:
    """Save a complete experiment manifest for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": experiment_name or "unnamed",
        "config_hash": compute_config_hash(config),
        "config": config,
        "environment": get_environment_info(),
        "results_summary": {
            k: v
            for k, v in results.items()
            if k
            in ["embedding_metrics", "llm_metrics", "rl_budget_metrics", "elapsed_s"]
        },
        "saved_at": datetime.now().isoformat(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment_name or "experiment"
    filepath = output_dir / f"manifest_{name}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Experiment manifest saved to {filepath}")
    return filepath


def load_experiment_manifest(filepath: Path) -> dict:
    """Load a previously saved experiment manifest."""
    with open(filepath) as f:
        return json.load(f)
