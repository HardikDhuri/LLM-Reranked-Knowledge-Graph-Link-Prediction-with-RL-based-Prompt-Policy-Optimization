"""Integration tests for utils modules working together."""

from __future__ import annotations

import json
import random

import numpy as np

from src.utils.cost_tracker import CostTracker
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import (
    compute_config_hash,
    load_experiment_manifest,
    save_experiment_manifest,
    set_all_seeds,
)

# ---------------------------------------------------------------------------
# setup_logging + CostTracker + set_all_seeds work together
# ---------------------------------------------------------------------------


def test_setup_logging_does_not_interfere_with_cost_tracker(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    logger = setup_logging(level="WARNING", log_to_console=False, log_dir=str(tmp_path))
    assert logger is not None

    tracker = CostTracker(model="gpt-4o-mini")
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        latency_s=0.1,
    )
    assert tracker.total_calls == 1


def test_set_all_seeds_then_cost_tracker_deterministic():
    set_all_seeds(42)
    tracker = CostTracker(model="default")
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_s=0.5,
    )
    assert tracker.total_cost_usd > 0.0
    assert tracker.total_calls == 1


def test_set_all_seeds_makes_random_deterministic():
    set_all_seeds(99)
    a = [random.random() for _ in range(5)]
    set_all_seeds(99)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_all_seeds_makes_numpy_deterministic():
    set_all_seeds(77)
    a = np.random.rand(5).tolist()
    set_all_seeds(77)
    b = np.random.rand(5).tolist()
    assert a == b


# ---------------------------------------------------------------------------
# save_experiment_manifest + load_experiment_manifest roundtrip
# ---------------------------------------------------------------------------


def test_save_load_manifest_roundtrip(tmp_path):
    config = {"seed": 42, "num_candidates": 50, "model": "gpt-4o-mini"}
    results = {
        "embedding_metrics": {"MRR": 0.45},
        "llm_metrics": {"MRR": 0.60},
    }
    filepath = save_experiment_manifest(
        config=config,
        results=results,
        output_dir=tmp_path,
        experiment_name="roundtrip_test",
    )
    assert filepath.exists()

    loaded = load_experiment_manifest(filepath)
    assert loaded["config"] == config
    assert loaded["results_summary"]["embedding_metrics"] == {"MRR": 0.45}
    assert loaded["results_summary"]["llm_metrics"] == {"MRR": 0.60}
    assert loaded["experiment_name"] == "roundtrip_test"


def test_save_manifest_includes_config_hash(tmp_path):
    config = {"seed": 1, "lr": 0.01}
    filepath = save_experiment_manifest(config=config, results={}, output_dir=tmp_path)
    loaded = load_experiment_manifest(filepath)
    expected_hash = compute_config_hash(config)
    assert loaded["config_hash"] == expected_hash


def test_save_manifest_without_experiment_name(tmp_path):
    filepath = save_experiment_manifest(config={}, results={}, output_dir=tmp_path)
    loaded = load_experiment_manifest(filepath)
    assert loaded["experiment_name"] == "unnamed"


def test_save_manifest_results_summary_filtered(tmp_path):
    """Only whitelisted keys should appear in results_summary."""
    results = {
        "embedding_metrics": {"MRR": 0.5},
        "llm_metrics": {"MRR": 0.6},
        "rl_budget_metrics": {"MRR": 0.55},
        "elapsed_s": 3.14,
        "should_be_excluded": "secret",
    }
    filepath = save_experiment_manifest(config={}, results=results, output_dir=tmp_path)
    loaded = load_experiment_manifest(filepath)
    summary = loaded["results_summary"]
    assert "embedding_metrics" in summary
    assert "llm_metrics" in summary
    assert "rl_budget_metrics" in summary
    assert "elapsed_s" in summary
    assert "should_be_excluded" not in summary


# ---------------------------------------------------------------------------
# CostTracker.save() + reload produces valid data
# ---------------------------------------------------------------------------


def test_cost_tracker_save_reload(tmp_path):
    tracker = CostTracker(model="gpt-4o-mini")
    tracker.record_call(
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        latency_s=0.2,
        query_info="test query",
        template_id="minimal",
    )

    output_file = tmp_path / "costs.json"
    tracker.save(output_file)

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert "records" in data
    assert len(data["records"]) == 1
    record = data["records"][0]
    assert record["model"] == "gpt-4o-mini"
    assert record["total_tokens"] == 30
    assert record["template_id"] == "minimal"


def test_cost_tracker_save_multiple_records(tmp_path):
    tracker = CostTracker(model="default")
    for i in range(3):
        tracker.record_call(
            usage={
                "prompt_tokens": 10 * i,
                "completion_tokens": 5,
                "total_tokens": 10 * i + 5,
            },
            latency_s=0.1,
        )

    output_file = tmp_path / "costs.json"
    tracker.save(output_file)

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data["records"]) == 3


def test_cost_tracker_total_cost_after_save(tmp_path):
    tracker = CostTracker(model="gpt-4o")
    tracker.record_call(
        usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        latency_s=0.5,
    )

    total_before = tracker.total_cost_usd
    output_file = tmp_path / "costs.json"
    tracker.save(output_file)

    # Stats should remain unchanged after save
    assert tracker.total_cost_usd == total_before
    assert tracker.total_calls == 1


# ---------------------------------------------------------------------------
# Compute config hash consistency
# ---------------------------------------------------------------------------


def test_config_hash_consistent_across_calls():
    config = {"a": 1, "b": [1, 2], "c": {"nested": True}}
    h1 = compute_config_hash(config)
    h2 = compute_config_hash(config)
    assert h1 == h2


def test_config_hash_changes_with_different_config():
    h1 = compute_config_hash({"x": 1})
    h2 = compute_config_hash({"x": 2})
    assert h1 != h2
