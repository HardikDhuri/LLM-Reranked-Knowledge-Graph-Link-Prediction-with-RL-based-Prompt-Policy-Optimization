"""Tests for src/analysis/results_loader.py."""

from __future__ import annotations

import json

from src.analysis.results_loader import ResultsLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _make_experiment(timestamp="20240101_120000"):
    return {
        "timestamp": timestamp,
        "config": {"template_id": "minimal"},
        "embedding_metrics": {"MRR": 0.5, "Hits@1": 0.3},
    }


def _make_ablation(timestamp="20240101_130000"):
    return {
        "timestamp": timestamp,
        "config": {"dims": [32, 64]},
        "results": [
            {"dim": 32, "metrics": {"MRR": 0.4}},
            {"dim": 64, "metrics": {"MRR": 0.5}},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_experiments_finds_json_files(tmp_path):
    _write_json(tmp_path / "experiment_20240101_120000.json", _make_experiment())
    _write_json(
        tmp_path / "ablation_embedding_dim_20240101_130000.json",
        _make_ablation(),
    )

    loader = ResultsLoader(results_dir=str(tmp_path))
    experiments = loader.list_experiments()

    assert len(experiments) == 2
    filenames = [e["filename"] for e in experiments]
    assert "experiment_20240101_120000.json" in filenames
    assert "ablation_embedding_dim_20240101_130000.json" in filenames


def test_list_experiments_empty_dir(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    experiments = loader.list_experiments()
    assert experiments == []


def test_list_experiments_skips_invalid_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json")
    _write_json(tmp_path / "experiment_20240101_120000.json", _make_experiment())

    loader = ResultsLoader(results_dir=str(tmp_path))
    experiments = loader.list_experiments()
    assert len(experiments) == 1


def test_load_experiment_returns_parsed_dict(tmp_path):
    data = _make_experiment()
    _write_json(tmp_path / "experiment_20240101_120000.json", data)

    loader = ResultsLoader(results_dir=str(tmp_path))
    result = loader.load_experiment("experiment_20240101_120000.json")

    assert result["timestamp"] == "20240101_120000"
    assert "embedding_metrics" in result


def test_load_latest_returns_most_recent(tmp_path):
    _write_json(
        tmp_path / "experiment_20240101_120000.json",
        _make_experiment("20240101_120000"),
    )
    _write_json(
        tmp_path / "experiment_20240102_120000.json",
        _make_experiment("20240102_120000"),
    )

    loader = ResultsLoader(results_dir=str(tmp_path))
    latest = loader.load_latest("experiment")

    assert latest["timestamp"] == "20240102_120000"


def test_load_latest_no_type_filter_returns_most_recent(tmp_path):
    _write_json(
        tmp_path / "experiment_20240101_120000.json",
        _make_experiment("20240101_120000"),
    )
    _write_json(
        tmp_path / "ablation_embedding_dim_20240103_120000.json",
        _make_ablation("20240103_120000"),
    )

    loader = ResultsLoader(results_dir=str(tmp_path))
    latest = loader.load_latest()

    assert latest["timestamp"] == "20240103_120000"


def test_load_latest_returns_none_for_empty_dir(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader.load_latest("experiment") is None


def test_load_ablation_finds_ablation_files(tmp_path):
    _write_json(
        tmp_path / "ablation_embedding_dim_20240101_120000.json", _make_ablation()
    )
    _write_json(
        tmp_path / "ablation_embedding_dim_20240102_120000.json", _make_ablation()
    )
    _write_json(tmp_path / "experiment_20240101_120000.json", _make_experiment())

    loader = ResultsLoader(results_dir=str(tmp_path))
    ablation_results = loader.load_ablation("embedding_dim")

    assert len(ablation_results) == 2


def test_load_ablation_returns_empty_for_missing(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader.load_ablation("nonexistent") == []


def test_infer_type_ablation(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader._infer_type("ablation_embedding_dim_20240101.json") == "ablation"


def test_infer_type_budget(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader._infer_type("budget_experiment_20240101.json") == "budget"


def test_infer_type_experiment(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader._infer_type("experiment_20240101.json") == "experiment"


def test_infer_type_cost(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader._infer_type("cost_report_20240101.json") == "cost"


def test_infer_type_other(tmp_path):
    loader = ResultsLoader(results_dir=str(tmp_path))
    assert loader._infer_type("random_stuff_20240101.json") == "other"


def test_list_experiments_includes_metadata(tmp_path):
    data = _make_experiment("20240101_120000")
    _write_json(tmp_path / "experiment_20240101_120000.json", data)

    loader = ResultsLoader(results_dir=str(tmp_path))
    experiments = loader.list_experiments()

    assert len(experiments) == 1
    e = experiments[0]
    assert e["filename"] == "experiment_20240101_120000.json"
    assert e["timestamp"] == "20240101_120000"
    assert e["type"] == "experiment"
    assert "config" in e
