"""LLM cost tracking utilities."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Approximate token costs (USD per 1K tokens) for common models
DEFAULT_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""

    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    estimated_cost_usd: float
    query_info: str = ""
    template_id: str = ""
    success: bool = True


class CostTracker:
    """Tracks LLM API costs across an experiment."""

    def __init__(self, model: str = "default", pricing: dict = None):
        self.model = model
        self.pricing = pricing or DEFAULT_PRICING
        self.records: list[LLMCallRecord] = []
        self.start_time = time.time()

    def _get_pricing(self) -> dict[str, float]:
        """Get pricing for the current model."""
        return self.pricing.get(self.model, self.pricing["default"])

    def record_call(
        self,
        usage: dict,
        latency_s: float,
        query_info: str = "",
        template_id: str = "",
        success: bool = True,
    ):
        """Record an LLM call with usage stats."""
        pricing = self._get_pricing()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        cost = (
            prompt_tokens / 1000 * pricing["input"]
            + completion_tokens / 1000 * pricing["output"]
        )

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=round(latency_s, 4),
            estimated_cost_usd=round(cost, 6),
            query_info=query_info,
            template_id=template_id,
            success=success,
        )
        self.records.append(record)
        return record

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def successful_calls(self) -> int:
        return sum(1 for r in self.records if r.success)

    @property
    def failed_calls(self) -> int:
        return sum(1 for r in self.records if not r.success)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.records)

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self.records)

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / max(1, self.total_calls)

    @property
    def avg_cost_per_call(self) -> float:
        return self.total_cost_usd / max(1, self.total_calls)

    def cost_by_template(self) -> dict[str, dict]:
        """Aggregate costs by template ID."""
        by_template: dict[str, dict] = {}
        for r in self.records:
            tid = r.template_id or "unknown"
            if tid not in by_template:
                by_template[tid] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            by_template[tid]["calls"] += 1
            by_template[tid]["tokens"] += r.total_tokens
            by_template[tid]["cost_usd"] += r.estimated_cost_usd
        # Round costs
        for tid in by_template:
            by_template[tid]["cost_usd"] = round(by_template[tid]["cost_usd"], 6)
        return by_template

    def summary(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "model": self.model,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_call_usd": round(self.avg_cost_per_call, 6),
            "total_latency_s": round(self.total_latency_s, 2),
            "avg_latency_s": round(self.avg_latency_s, 4),
            "wall_time_s": round(elapsed, 2),
            "cost_by_template": self.cost_by_template(),
        }

    def save(self, filepath: Path):
        """Save cost records and summary to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cost report saved to {filepath}")

    def print_summary(self):
        """Print a formatted cost summary."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("LLM COST SUMMARY")
        print("=" * 50)
        print(f"Model:               {s['model']}")
        print(
            f"Total calls:         {s['total_calls']} "
            f"({s['successful_calls']} ok, {s['failed_calls']} failed)"
        )
        print(
            f"Total tokens:        {s['total_tokens']:,} "
            f"(prompt: {s['total_prompt_tokens']:,}, "
            f"completion: {s['total_completion_tokens']:,})"
        )
        print(f"Estimated cost:      ${s['total_cost_usd']:.4f}")
        print(f"Avg cost/call:       ${s['avg_cost_per_call_usd']:.6f}")
        print(f"Total latency:       {s['total_latency_s']:.1f}s")
        print(f"Avg latency/call:    {s['avg_latency_s']:.3f}s")
        print(f"Wall time:           {s['wall_time_s']:.1f}s")
        if s["cost_by_template"]:
            print("\nCost by template:")
            for tid, info in s["cost_by_template"].items():
                print(
                    f"  {tid}: {info['calls']} calls, "
                    f"{info['tokens']} tokens, ${info['cost_usd']:.4f}"
                )
        print("=" * 50)
