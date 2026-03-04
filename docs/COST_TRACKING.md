# Cost Tracking

This document explains how the pipeline tracks and reports LLM API costs.

## How Cost Tracking Works

Every call to the LLM API made through `src/models/llm_client.py` is recorded by `src/utils/cost_tracker.py`. The tracker stores:

- Timestamp of the call
- Model name
- Prompt token count
- Completion token count
- Total token count
- Latency in seconds
- Estimated cost in USD
- Associated query information
- Template ID used
- Success/failure status

The `CostTracker` instance is passed through the evaluation pipeline and populated automatically.

## Pricing Model

Costs are estimated using approximate per-token prices (USD per 1,000 tokens):

| Model | Input (prompt) | Output (completion) |
|---|---|---|
| `gpt-4o` | $0.005 | $0.015 |
| `gpt-4o-mini` | $0.00015 | $0.0006 |
| `gpt-4-turbo` | $0.010 | $0.030 |
| `gpt-3.5-turbo` | $0.0005 | $0.0015 |
| *(default)* | $0.001 | $0.002 |

Pricing is defined in `DEFAULT_PRICING` in `src/utils/cost_tracker.py` and can be overridden when constructing the tracker:

```python
from src.utils.cost_tracker import CostTracker

tracker = CostTracker(
    model="gpt-4o-mini",
    pricing={
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    },
)
```

> **Note:** These prices are estimates based on OpenAI's public pricing at the time of writing. Actual costs may differ. Check the provider's pricing page for current rates.

## Cost Formula

$$\text{cost} = \frac{n_{\text{prompt}}}{1000} \times p_{\text{input}} + \frac{n_{\text{completion}}}{1000} \times p_{\text{output}}$$

where $n_{\text{prompt}}$ and $n_{\text{completion}}$ are token counts and $p_{\text{input}}$, $p_{\text{output}}$ are per-1K-token prices.

## Generating Cost Reports

### During an Experiment

The experiment runner automatically prints a cost summary at the end:

```
==================================================
LLM COST SUMMARY
==================================================
Model:               gpt-4o-mini
Total calls:         75 (75 ok, 0 failed)
Total tokens:        18,432 (prompt: 14,210, completion: 4,222)
Estimated cost:      $0.0053
Avg cost/call:       $0.000071
Total latency:       42.3s
Avg latency/call:    0.564s
Wall time:           45.1s

Cost by template:
  minimal: 25 calls, 4,150 tokens, $0.0011
  with_descriptions: 25 calls, 7,080 tokens, $0.0024
  strict_rubric: 25 calls, 7,202 tokens, $0.0018
==================================================
```

### Saving a Cost Report

```python
tracker.save(Path("results/cost_report.json"))
```

The JSON report contains a `summary` object and a `records` list:

```json
{
  "summary": {
    "model": "gpt-4o-mini",
    "total_calls": 75,
    "successful_calls": 75,
    "failed_calls": 0,
    "total_tokens": 18432,
    "total_cost_usd": 0.005312,
    "cost_by_template": {
      "minimal": {"calls": 25, "tokens": 4150, "cost_usd": 0.001123},
      "with_descriptions": {"calls": 25, "tokens": 7080, "cost_usd": 0.002401}
    }
  },
  "records": [
    {
      "timestamp": "2024-01-15T10:23:45.123456",
      "model": "gpt-4o-mini",
      "prompt_tokens": 142,
      "completion_tokens": 48,
      "total_tokens": 190,
      "latency_s": 0.512,
      "estimated_cost_usd": 0.000050,
      "query_info": "(Barack Obama, /people/person/nationality, ?)",
      "template_id": "minimal",
      "success": true
    }
  ]
}
```

### Standalone Cost Report Script

```bash
python -m scripts.cost_report --results-dir results/
```

## Cost by Template

The `cost_by_template()` method aggregates costs across all calls grouped by template ID. This is useful for comparing the cost-effectiveness of different prompt templates:

```python
breakdown = tracker.cost_by_template()
# {
#   "minimal":          {"calls": 25, "tokens": 4150,  "cost_usd": 0.0011},
#   "with_descriptions": {"calls": 25, "tokens": 7080, "cost_usd": 0.0024},
#   "strict_rubric":     {"calls": 25, "tokens": 7202, "cost_usd": 0.0018},
# }
```

Shorter templates like `minimal` are typically 2–3× cheaper per call than richer templates like `with_descriptions` or `strict_rubric`.

## Budget Experiment Interpretation

The budget experiment (`src/rl/budget_experiment.py`, `scripts/budget_demo.py`) compares three strategies under a fixed LLM call budget:

| Strategy | Description |
|---|---|
| `all_embedding` | Never call the LLM; use embedding ranking only |
| `all_llm` | Always call the LLM (requires budget ≥ num_queries) |
| `rl_agent` | Use the RL agent to decide which queries get LLM calls |

Key output fields:

```json
{
  "budget_fraction": 0.5,
  "all_embedding_mrr": 0.182,
  "all_llm_mrr": 0.241,
  "rl_agent_mrr": 0.228,
  "llm_fraction_used": 0.50,
  "budget_efficiency": 0.82
}
```

- **`budget_efficiency`** = `(rl_agent_mrr - all_embedding_mrr) / (all_llm_mrr - all_embedding_mrr)` — how much of the LLM gain the RL agent captures at fraction of the cost.
- A well-tuned agent achieves `budget_efficiency > 0.8` with only 50% of the budget.

### Running Budget Sweeps

```bash
# Sweep budget from 10% to 100% of queries
python -m scripts.budget_sweep --num-queries 100 --budgets 10,20,30,50,70,100
```

Results are saved to `results/budget_sweep_<timestamp>.json` and can be visualised with notebook `07_budget_experiment.ipynb`.
