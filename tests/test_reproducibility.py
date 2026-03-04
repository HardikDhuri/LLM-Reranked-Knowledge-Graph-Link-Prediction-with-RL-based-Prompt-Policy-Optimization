"""Tests for src/utils/reproducibility.py."""

import json
import random

import numpy as np

from src.utils.reproducibility import (
    compute_config_hash,
    get_environment_info,
    load_experiment_manifest,
    save_experiment_manifest,
    set_all_seeds,
)

# ---------------------------------------------------------------------------
# set_all_seeds
# ---------------------------------------------------------------------------


def test_set_all_seeds_deterministic_random():
    set_all_seeds(42)
    vals1 = [random.random() for _ in range(5)]
    set_all_seeds(42)
    vals2 = [random.random() for _ in range(5)]
    assert vals1 == vals2


def test_set_all_seeds_deterministic_numpy():
    set_all_seeds(123)
    arr1 = np.random.rand(5).tolist()
    set_all_seeds(123)
    arr2 = np.random.rand(5).tolist()
    assert arr1 == arr2


def test_set_all_seeds_different_seeds_differ():
    set_all_seeds(1)
    vals1 = [random.random() for _ in range(5)]
    set_all_seeds(2)
    vals2 = [random.random() for _ in range(5)]
    assert vals1 != vals2


# ---------------------------------------------------------------------------
# get_environment_info
# ---------------------------------------------------------------------------


def test_get_environment_info_returns_dict():
    info = get_environment_info()
    assert isinstance(info, dict)


def test_get_environment_info_expected_keys():
    info = get_environment_info()
    for key in ("platform", "python_version", "timestamp"):
        assert key in info, f"Missing key: {key}"


def test_get_environment_info_has_torch_version():
    info = get_environment_info()
    assert "torch_version" in info


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


def test_compute_config_hash_same_config():
    config = {"model": "gpt-4o", "seed": 42, "topk": 5}
    h1 = compute_config_hash(config)
    h2 = compute_config_hash(config)
    assert h1 == h2


def test_compute_config_hash_different_configs():
    config1 = {"model": "gpt-4o", "seed": 42}
    config2 = {"model": "gpt-4o-mini", "seed": 42}
    assert compute_config_hash(config1) != compute_config_hash(config2)


def test_compute_config_hash_key_order_independent():
    config_a = {"a": 1, "b": 2}
    config_b = {"b": 2, "a": 1}
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_compute_config_hash_length():
    h = compute_config_hash({"x": 1})
    assert len(h) == 12


# ---------------------------------------------------------------------------
# save_experiment_manifest / load_experiment_manifest
# ---------------------------------------------------------------------------


def test_save_experiment_manifest_creates_file(tmp_path):
    config = {"model": "gpt-4o", "seed": 42}
    results = {"embedding_metrics": {"mrr": 0.5}, "elapsed_s": 10.0}
    filepath = save_experiment_manifest(
        config, results, tmp_path, experiment_name="test_run"
    )
    assert filepath.exists()


def test_save_experiment_manifest_valid_json(tmp_path):
    config = {"model": "gpt-4o", "seed": 42}
    results = {"embedding_metrics": {"mrr": 0.5}, "elapsed_s": 10.0}
    filepath = save_experiment_manifest(config, results, tmp_path)
    with open(filepath) as f:
        data = json.load(f)
    assert "experiment_name" in data
    assert "config_hash" in data
    assert "config" in data
    assert "environment" in data
    assert "results_summary" in data
    assert "saved_at" in data


def test_save_experiment_manifest_config_hash_matches(tmp_path):
    config = {"model": "gpt-4o", "seed": 99}
    results = {}
    filepath = save_experiment_manifest(config, results, tmp_path)
    with open(filepath) as f:
        data = json.load(f)
    assert data["config_hash"] == compute_config_hash(config)


def test_load_experiment_manifest_roundtrip(tmp_path):
    config = {"model": "gpt-3.5-turbo", "seed": 7}
    results = {"elapsed_s": 5.0}
    filepath = save_experiment_manifest(
        config, results, tmp_path, experiment_name="roundtrip"
    )
    loaded = load_experiment_manifest(filepath)
    assert loaded["config"] == config
    assert loaded["experiment_name"] == "roundtrip"


def test_save_experiment_manifest_results_summary_filtered(tmp_path):
    config = {"seed": 1}
    results = {
        "embedding_metrics": {"mrr": 0.3},
        "llm_metrics": {"hits@1": 0.4},
        "rl_budget_metrics": {"reward": 1.0},
        "elapsed_s": 20.0,
        "extra_key": "should_not_appear",
    }
    filepath = save_experiment_manifest(config, results, tmp_path)
    with open(filepath) as f:
        data = json.load(f)
    summary = data["results_summary"]
    assert "embedding_metrics" in summary
    assert "llm_metrics" in summary
    assert "rl_budget_metrics" in summary
    assert "elapsed_s" in summary
    assert "extra_key" not in summary
