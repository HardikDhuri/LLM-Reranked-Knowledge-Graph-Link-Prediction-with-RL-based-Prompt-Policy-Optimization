# RL Agents

This document describes the reinforcement learning agents used for prompt selection and budget allocation.

## Overview

The pipeline uses two contextual bandit agents:

| Agent | Module | Arms | Reward |
|---|---|---|---|
| `RLPromptSelector` | `src/rl/prompt_selector.py` | 5 (one per prompt template) | Reciprocal rank of true entity |
| `BudgetAgent` | `src/rl/budget_agent.py` | 2 (embedding-only vs LLM) | Reciprocal rank of true entity |

Both agents are backed by either `LinUCBAgent` or `EpsilonGreedyAgent` from `src/rl/bandit.py`.

---

## LinUCB Algorithm

`LinUCBAgent` (`src/rl/bandit.py`) implements the **Linear Upper Confidence Bound** algorithm (Li et al., 2010).

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_arms` | varies | Number of arms (templates or budget options) |
| `feature_dim` | 8 | Dimension of the context vector |
| `alpha` | 1.0 | Exploration parameter (higher → more exploration) |

### State

Per arm $a$, the agent maintains:
- $A_a \in \mathbb{R}^{d \times d}$ — regularised feature covariance matrix (initialised to $I_d$)
- $b_a \in \mathbb{R}^d$ — weighted reward accumulator (initialised to $\mathbf{0}$)

### Arm Selection

For context vector $\mathbf{x} \in \mathbb{R}^d$, select:

$$a^* = \arg\max_{a} \left[ \hat{\theta}_a^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top A_a^{-1} \mathbf{x}} \right]$$

where $\hat{\theta}_a = A_a^{-1} b_a$ is the estimated reward weight vector and the square-root term is the upper confidence bonus.

### Update Rule

After observing reward $r$ for arm $a$ with context $\mathbf{x}$:

$$A_a \leftarrow A_a + \mathbf{x}\mathbf{x}^\top$$
$$b_a \leftarrow b_a + r \cdot \mathbf{x}$$

This is equivalent to ridge regression with regularisation $\lambda = 1$.

### Exploration–Exploitation Trade-off

- **Low alpha (e.g. 0.1):** Exploits known best arms; converges quickly but may miss better options.
- **High alpha (e.g. 2.0):** Explores more; slower convergence but better asymptotic performance.
- **alpha = 1.0 (default):** Balanced starting point.

---

## Epsilon-Greedy Baseline

`EpsilonGreedyAgent` (`src/rl/bandit.py`) is a simple non-contextual baseline that ignores the feature vector.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_arms` | varies | Number of arms |
| `epsilon` | 0.1 | Exploration probability |

### Selection Rule

With probability $\epsilon$: pick a random arm (exploration).
With probability $1 - \epsilon$: pick the arm with the highest average reward so far (exploitation).

During the first $n\_\text{arms}$ steps, the agent always explores to initialise reward estimates.

---

## Feature Extraction

`QueryFeatureExtractor` (`src/rl/features.py`) converts a query triple `(h, r, t)` into an 8-dimensional feature vector that is used as the bandit context.

Features are computed from entity and relation frequency statistics derived from the training triples.

| Index | Feature | Description |
|---|---|---|
| 0 | `h_freq` | Normalised head entity frequency |
| 1 | `t_freq` | Normalised tail entity frequency |
| 2 | `r_freq` | Normalised relation frequency |
| 3 | `h_rank` | Head entity frequency rank (normalised, 0 = most frequent) |
| 4 | `t_rank` | Tail entity frequency rank (normalised) |
| 5 | `r_rank` | Relation frequency rank (normalised) |
| 6 | `h_is_frequent` | 1 if head frequency > median entity frequency, else 0 |
| 7 | `t_is_frequent` | 1 if tail frequency > median entity frequency, else 0 |

All values are in $[0, 1]$.

**Rationale:** High-frequency entities and relations tend to have more factual information available to the LLM, making richer prompt templates (e.g. `with_descriptions`, `strict_rubric`) more beneficial. Low-frequency entities may have sparse Wikidata descriptions, making minimal templates more reliable.

---

## RL Prompt Selector

`RLPromptSelector` (`src/rl/prompt_selector.py`) wraps either `LinUCBAgent` or `EpsilonGreedyAgent` to select a prompt template for each query.

### Configuration

```python
from src.rl.prompt_selector import RLPromptSelector

selector = RLPromptSelector(
    template_ids=["minimal", "with_descriptions", "strict_rubric", "concise_cot", "binary_judge"],
    feature_extractor=extractor,
    agent_type="linucb",   # or "epsilon_greedy"
    alpha=1.0,
    epsilon=0.1,
)
```

### Usage

```python
# Select template for a query
template_id = selector.select(h="Barack Obama", r="/people/person/nationality", t="?")

# After evaluation, update with observed reward
selector.update(
    h="Barack Obama",
    r="/people/person/nationality",
    t="United States of America",
    template_id=template_id,
    reward=1.0 / rank,
)
```

---

## Budget Allocation Agent

`BudgetAgent` (`src/rl/budget_agent.py`) decides whether to invoke the expensive LLM reranker or use embedding-only scoring for each query, under a fixed call budget.

### Arms

| Arm | Name | Cost |
|---|---|---|
| 0 | `embedding_only` | Free |
| 1 | `llm_reranker` | 1 budget unit per call |

### Budget Enforcement

If `remaining_budget < cost_per_llm_query`, the agent always returns `ARM_EMBEDDING` regardless of what the bandit would select. This ensures the budget is never exceeded.

### Configuration

```python
from src.rl.budget_agent import BudgetAgent

agent = BudgetAgent(
    feature_extractor=extractor,
    total_budget=50,        # maximum LLM calls
    agent_type="linucb",    # or "epsilon_greedy"
    alpha=1.0,
    cost_per_llm_query=1,
)
```

### Usage

```python
# Decide for a query
action = agent.decide(h, r, t)

# After evaluation
agent.record_decision(
    h, r, t,
    action=action,
    reward=1.0 / rank,
    embedding_rank=embed_rank,
    llm_rank=llm_rank,
)

# Get summary
print(agent.summary())
# {"total_budget": 50, "remaining_budget": 23, "llm_fraction": 0.54, ...}
```

---

## Reward Signal

Both agents use **reciprocal rank** as the reward signal:

$$r = \frac{1}{\text{rank}(t_{\text{true}})}$$

- Rank 1 (true answer is first) → reward = 1.0
- Rank 5 → reward = 0.2
- Rank 25 (last in top-K) → reward = 0.04

This signal directly optimises for MRR, which is the primary evaluation metric.

---

## Hyperparameter Tuning Guide

### LinUCB `alpha`

- Start with `alpha=1.0`.
- If the agent converges too slowly (keeps exploring suboptimal arms): decrease `alpha`.
- If the agent gets stuck on a suboptimal arm early: increase `alpha`.
- Typical range: 0.1 – 3.0.

### EpsilonGreedy `epsilon`

- `epsilon=0.1` (10% exploration) is a standard starting point.
- For very few queries (< 50): use `epsilon=0.3` to ensure sufficient exploration.
- For long runs (> 1000 queries): consider decaying epsilon over time.

### Budget Agent `total_budget`

- Set to the maximum number of LLM API calls you can afford per evaluation run.
- A budget of ~50% of total queries (e.g. `total_budget=50` for 100 queries) gives the agent meaningful decisions.
- With `total_budget >= num_queries`, the budget constraint is never binding and the agent always uses the LLM.

### Running Budget Experiments

```bash
# Sweep over budget fractions
python -m scripts.budget_sweep --num-queries 100

# Single budget experiment
python -m scripts.budget_demo --budget 20 --num-queries 50 --agent-type linucb
```
