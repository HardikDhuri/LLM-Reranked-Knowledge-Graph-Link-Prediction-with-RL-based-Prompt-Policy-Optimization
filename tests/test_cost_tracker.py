"""Tests for src/utils/cost_tracker.py."""

import json

import pytest

from src.utils.cost_tracker import DEFAULT_PRICING, CostTracker, LLMCallRecord

# ---------------------------------------------------------------------------
# record_call
# ---------------------------------------------------------------------------


def test_record_call_creates_record():
    tracker = CostTracker(model="gpt-4o-mini")
    record = tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_s=0.5,
        query_info="test query",
        template_id="tmpl1",
    )
    assert isinstance(record, LLMCallRecord)
    assert record.prompt_tokens == 100
    assert record.completion_tokens == 50
    assert record.total_tokens == 150
    assert record.latency_s == 0.5
    assert record.query_info == "test query"
    assert record.template_id == "tmpl1"
    assert record.success is True


def test_record_call_computes_cost():
    tracker = CostTracker(model="gpt-4o-mini")
    record = tracker.record_call(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        latency_s=1.0,
    )
    expected = (
        1000 / 1000 * DEFAULT_PRICING["gpt-4o-mini"]["input"]
        + 1000 / 1000 * DEFAULT_PRICING["gpt-4o-mini"]["output"]
    )
    assert record.estimated_cost_usd == pytest.approx(expected, rel=1e-5)


def test_record_call_total_tokens_computed_if_missing():
    tracker = CostTracker(model="default")
    record = tracker.record_call(
        usage={"prompt_tokens": 40, "completion_tokens": 20},
        latency_s=0.1,
    )
    assert record.total_tokens == 60


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_total_calls():
    tracker = CostTracker()
    assert tracker.total_calls == 0
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5}, latency_s=0.1
    )
    tracker.record_call(
        usage={"prompt_tokens": 20, "completion_tokens": 10}, latency_s=0.2
    )
    assert tracker.total_calls == 2


def test_successful_and_failed_calls():
    tracker = CostTracker()
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        latency_s=0.1,
        success=True,
    )
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        latency_s=0.1,
        success=False,
    )
    assert tracker.successful_calls == 1
    assert tracker.failed_calls == 1


def test_total_tokens():
    tracker = CostTracker()
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_s=0.1,
    )
    tracker.record_call(
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        latency_s=0.2,
    )
    assert tracker.total_tokens == 450


def test_total_cost_usd():
    tracker = CostTracker(model="default")
    tracker.record_call(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000}, latency_s=0.1
    )
    tracker.record_call(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000}, latency_s=0.1
    )
    pricing = DEFAULT_PRICING["default"]
    expected = 2 * (1.0 * pricing["input"] + 1.0 * pricing["output"])
    assert tracker.total_cost_usd == pytest.approx(expected, rel=1e-5)


def test_avg_latency_s():
    tracker = CostTracker()
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5}, latency_s=1.0
    )
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5}, latency_s=3.0
    )
    assert tracker.avg_latency_s == pytest.approx(2.0, rel=1e-5)


def test_avg_latency_s_no_calls():
    tracker = CostTracker()
    assert tracker.avg_latency_s == 0.0


# ---------------------------------------------------------------------------
# cost_by_template
# ---------------------------------------------------------------------------


def test_cost_by_template():
    tracker = CostTracker(model="default")
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_s=0.1,
        template_id="t1",
    )
    tracker.record_call(
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        latency_s=0.2,
        template_id="t1",
    )
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_s=0.1,
        template_id="t2",
    )

    by_template = tracker.cost_by_template()
    assert "t1" in by_template
    assert "t2" in by_template
    assert by_template["t1"]["calls"] == 2
    assert by_template["t1"]["tokens"] == 450
    assert by_template["t2"]["calls"] == 1
    assert by_template["t2"]["tokens"] == 150


def test_cost_by_template_unknown_template_id():
    tracker = CostTracker()
    tracker.record_call(
        usage={"prompt_tokens": 10, "completion_tokens": 5}, latency_s=0.1
    )
    by_template = tracker.cost_by_template()
    assert "unknown" in by_template


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_returns_expected_keys():
    tracker = CostTracker(model="gpt-4o")
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50}, latency_s=0.5
    )
    s = tracker.summary()
    expected_keys = [
        "model", "total_calls", "successful_calls", "failed_calls",
        "total_tokens", "total_prompt_tokens", "total_completion_tokens",
        "total_cost_usd", "avg_cost_per_call_usd", "total_latency_s",
        "avg_latency_s", "wall_time_s", "cost_by_template",
    ]
    for key in expected_keys:
        assert key in s, f"Missing key: {key}"


def test_summary_correct_values():
    tracker = CostTracker(model="default")
    tracker.record_call(
        usage={"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
        latency_s=2.0,
    )
    s = tracker.summary()
    assert s["total_calls"] == 1
    assert s["total_tokens"] == 1000
    assert s["total_latency_s"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_writes_valid_json(tmp_path):
    tracker = CostTracker(model="gpt-3.5-turbo")
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        latency_s=0.3,
        template_id="t1",
    )
    filepath = tmp_path / "cost_report.json"
    tracker.save(filepath)

    assert filepath.exists()
    with open(filepath) as f:
        data = json.load(f)
    assert "summary" in data
    assert "records" in data
    assert len(data["records"]) == 1


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


def test_print_summary_doesnt_crash(capsys):
    tracker = CostTracker(model="gpt-4o")
    tracker.record_call(
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        latency_s=0.5,
        template_id="t1",
    )
    tracker.print_summary()
    captured = capsys.readouterr()
    assert "LLM COST SUMMARY" in captured.out


# ---------------------------------------------------------------------------
# Custom pricing
# ---------------------------------------------------------------------------


def test_custom_pricing_overrides_default():
    custom_pricing = {
        "mymodel": {"input": 0.1, "output": 0.2},
        "default": {"input": 0.001, "output": 0.002},
    }
    tracker = CostTracker(model="mymodel", pricing=custom_pricing)
    record = tracker.record_call(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        latency_s=0.1,
    )
    expected = 1.0 * 0.1 + 1.0 * 0.2
    assert record.estimated_cost_usd == pytest.approx(expected, rel=1e-5)
