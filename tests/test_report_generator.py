"""Tests for src/analysis/report_generator.py."""

from __future__ import annotations

import json

from src.analysis.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _make_experiment_result(timestamp="20240101_120000"):
    return {
        "timestamp": timestamp,
        "config": {"template_id": "minimal", "num_test_queries": 10},
        "embedding_metrics": {"MRR": 0.5, "Hits@1": 0.3, "Hits@3": 0.5, "Hits@10": 0.7},
        "llm_metrics": {"MRR": 0.6, "Hits@1": 0.4, "Hits@3": 0.6, "Hits@10": 0.8},
        "llm_reranker_stats": {"total_scored": 100},
    }


def _make_ablation_result():
    return {
        "timestamp": "20240101_130000",
        "config": {"dims": [32, 64]},
        "results": [
            {"dim": 32, "metrics": {"MRR": 0.4, "Hits@1": 0.2}},
            {"dim": 64, "metrics": {"MRR": 0.5, "Hits@1": 0.3}},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_returns_string(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))
    assert isinstance(report, str)
    assert len(report) > 0


def test_generate_writes_file(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    out = tmp_path / "REPORT.md"
    gen.generate(output_path=str(out))
    assert out.exists()
    content = out.read_text()
    assert "# Experiment Report" in content


def test_generate_with_empty_results_dir(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))
    assert "No experiment results found" in report


def test_generate_with_experiment_results(tmp_path):
    _write_json(tmp_path / "experiment_20240101_120000.json", _make_experiment_result())
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))

    assert "experiment_20240101_120000.json" in report
    assert "Embedding Baseline" in report
    assert "LLM Reranker" in report


def test_generate_with_ablation_results(tmp_path):
    _write_json(
        tmp_path / "ablation_embedding_dim_20240101_130000.json",
        _make_ablation_result(),
    )
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))

    assert "Ablation Studies" in report
    assert "ablation_embedding_dim_20240101_130000" in report


def test_generate_contains_all_sections(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))

    assert "# Experiment Report" in report
    assert "## Overview" in report
    assert "## Conclusions" in report
    assert "report_generator.py" in report


def test_generate_default_output_path(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    gen.generate()
    assert (tmp_path / "REPORT.md").exists()


def test_format_metrics_table_valid_table(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    metrics = {"MRR": 0.5, "Hits@1": 0.3, "Hits@3": 0.5, "Hits@10": 0.7}
    table = gen._format_metrics_table(metrics)

    assert "| Metric | Value |" in table
    assert "| MRR |" in table
    assert "| Hits@1 |" in table
    assert "| Hits@3 |" in table
    assert "| Hits@10 |" in table
    assert "0.5000" in table


def test_format_metrics_table_missing_keys(tmp_path):
    gen = ReportGenerator(results_dir=str(tmp_path))
    metrics = {"MRR": 0.5}
    table = gen._format_metrics_table(metrics)

    assert "N/A" in table


def test_generate_cost_summary_with_llm_stats(tmp_path):
    _write_json(
        tmp_path / "experiment_20240101_120000.json",
        {
            "timestamp": "20240101_120000",
            "config": {},
            "llm_reranker_stats": {"total_scored": 42},
        },
    )
    gen = ReportGenerator(results_dir=str(tmp_path))
    report = gen.generate(output_path=str(tmp_path / "REPORT.md"))

    assert "Cost Summary" in report


def test_import_from_analysis_package():
    from src.analysis import ReportGenerator, ResultsLoader

    assert ReportGenerator is not None
    assert ResultsLoader is not None
