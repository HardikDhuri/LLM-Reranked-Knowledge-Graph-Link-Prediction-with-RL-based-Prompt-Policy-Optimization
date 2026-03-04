# Evaluation Protocol

This document describes how the pipeline evaluates link prediction performance.

## Filtered Ranking Setting

The pipeline uses the **filtered ranking** protocol, the standard for knowledge graph link prediction benchmarks (Bordes et al., 2013).

For a test query `(h, r, ?)`:
1. Score all entities in the entity vocabulary as the candidate tail.
2. Remove from the ranked list all entities `t'` such that `(h, r, t')` appears in the training, validation, *or* test set — **except** the query answer itself.
3. Determine the rank of the true answer `t` in the filtered list.

This avoids penalising the model for placing other *correct* answers above the query answer.

## Candidate Generation Strategy

Rather than scoring all ~15,000 FB15k-237 entities with the LLM (prohibitively expensive), the pipeline uses a **two-stage** approach:

```
Stage 1 (Embedding):  Score all entities with RotatE → keep Top-K (default K=25)
Stage 2 (LLM):        Score Top-K candidates with the LLM → rerank
```

The `src/eval/candidates.py` module implements Stage 1. It calls the RotatE embedding model to obtain scores for all entities, applies the filtered mask, and returns the top-K candidates with their embedding scores.

The `src/models/reranker.py` module implements Stage 2, calling the LLM scorer for each candidate and returning a re-sorted list.

> **Note:** When the RL budget agent decides to use embedding-only (`ARM_EMBEDDING`), Stage 2 is skipped and the embedding ranking is used directly.

## Metrics Definitions

All metrics are computed by `src/eval/metrics.py` over a list of `RankingResult` objects.

### Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $|Q|$ is the number of test queries and $\text{rank}_i$ is the filtered rank of the true answer for query $i$.

- Range: $(0, 1]$. Higher is better.
- A perfect model scores MRR = 1.0 (all true answers ranked first).

### Hits@K

$$\text{Hits@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{1}[\text{rank}_i \leq K]$$

- Computed for $K \in \{1, 3, 10\}$.
- Range: $[0, 1]$. Higher is better.
- Hits@1 is the fraction of queries where the true answer is ranked first.

### `compute_all_metrics` Output

```python
{
    "MRR":           float,   # mean reciprocal rank
    "Hits@1":        float,   # fraction ranked ≤ 1
    "Hits@3":        float,   # fraction ranked ≤ 3
    "Hits@10":       float,   # fraction ranked ≤ 10
    "num_queries":   int,     # number of evaluated queries
    "avg_candidates": float,  # average candidate list size
}
```

## How to Interpret Results

| Metric | Poor | Moderate | Good | Excellent |
|---|---|---|---|---|
| MRR | < 0.15 | 0.15–0.25 | 0.25–0.35 | > 0.35 |
| Hits@1 | < 0.10 | 0.10–0.20 | 0.20–0.30 | > 0.30 |
| Hits@10 | < 0.40 | 0.40–0.55 | 0.55–0.70 | > 0.70 |

These thresholds are approximate and based on published results on FB15k-237.

MRR and Hits@1 are the most informative metrics for real-world use cases where users expect the top-ranked answer to be correct. Hits@10 is useful for evaluating candidate recall.

## Expected Baselines

Published results on FB15k-237 (tail prediction, filtered):

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| RotatE (Sun et al., 2019) | 0.338 | 0.241 | 0.375 | 0.533 |
| TransE (Bordes et al., 2013) | 0.279 | 0.198 | 0.376 | 0.441 |
| ComplEx (Trouillon et al., 2016) | 0.247 | 0.158 | 0.275 | 0.428 |

> **Note:** The numbers above are from full-dataset evaluation. This pipeline sub-samples test queries (default: 15) for rapid iteration; results on small samples will vary significantly. Run with `SAMPLE_TEST_QUERIES=0` (or a large number) for full evaluation.

## Running Evaluation

```bash
# Evaluate the RotatE embedding baseline only
python -m scripts.eval_embedding

# Run the full LLM reranking evaluation
python -m scripts.run_experiment --template minimal --num-queries 100

# Compare all prompt templates
python -m scripts.compare_templates --num-queries 50
```

Results are saved as JSON to `$RESULTS_DIR/` with a timestamp.
