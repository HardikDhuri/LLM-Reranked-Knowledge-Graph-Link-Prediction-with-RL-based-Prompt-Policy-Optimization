# System Architecture

This document describes the architecture of the LLM-Reranked KG Link Prediction pipeline with RL-based prompt and budget optimisation.

## Overall System Architecture

```mermaid
flowchart TD
    subgraph Data
        A[FB15k-237\nTrain / Valid / Test]
    end

    subgraph Embedding
        B[PyKEEN RotatE\nembedding_baseline.py]
    end

    subgraph Grounding
        C[Wikidata SPARQL\nsparql.py]
        D[Disk Cache\ncache.py]
        C <--> D
    end

    subgraph LLM
        E[PromptTemplate\nrenderer.py]
        F[LLM Client\nllm_client.py]
        G[Triple Scorer\nscorer.py]
        H[Reranker\nreranker.py]
        E --> G --> F
        G --> H
    end

    subgraph RL
        I[QueryFeatureExtractor\nfeatures.py]
        J[LinUCBAgent\nbandit.py]
        K[RLPromptSelector\nprompt_selector.py]
        L[BudgetAgent\nbudget_agent.py]
        I --> J --> K
        I --> L
    end

    subgraph Evaluation
        M[CandidateGenerator\ncandidates.py]
        N[Evaluator\nevaluate.py]
        O[Metrics\nMRR / Hits@K]
        M --> N --> O
    end

    subgraph Utils
        P[CostTracker\ncost_tracker.py]
        Q[Reproducibility\nreproducibility.py]
        R[Logging\nlogging_config.py]
    end

    A --> B --> M
    M --> C
    C --> E
    K -->|chosen template| E
    L -->|embed or LLM| M
    H --> N
    F --> P
```

## Evaluation Pipeline

```mermaid
flowchart LR
    A[Test Query\nh, r, ?] --> B[Embedding Score\nall entities]
    B --> C[Top-K\nCandidates]
    C --> D[Wikidata\nGrounding]
    D --> E[Render Prompt\nfor each candidate]
    E --> F[LLM Score]
    F --> G[Sort by\nLLM Score]
    G --> H[Filtered\nRanking]
    H --> I[Reciprocal Rank\nHits@K]
```

**Filtered ranking** removes all known true triples (from train, valid, and test) from the candidate list before computing the rank of the query answer. This avoids penalising the model for correctly ranking other true triples above the query answer.

## RL Prompt Selection Loop

```mermaid
sequenceDiagram
    participant Q as Query (h, r, t)
    participant FE as FeatureExtractor
    participant LU as LinUCBAgent
    participant PT as PromptTemplate
    participant LLM as LLM API
    participant EV as Evaluator

    Q->>FE: extract(h, r, t)
    FE-->>LU: context vector x ∈ ℝ⁸
    LU-->>PT: select arm a = argmax UCB(x)
    PT-->>LLM: rendered prompt
    LLM-->>EV: plausibility scores
    EV-->>LU: reward r = 1 / rank(true_entity)
    LU->>LU: update A[a], b[a]
```

The LinUCB agent maintains per-arm ridge regression parameters `(A, b)` and selects the arm (prompt template) with the highest upper confidence bound:

```
UCB_a(x) = θ_a^T x + α √(x^T A_a^{-1} x)
```

## RL Budget Allocation Loop

```mermaid
sequenceDiagram
    participant Q as Query (h, r, t)
    participant FE as FeatureExtractor
    participant BA as BudgetAgent
    participant EM as Embedding Model
    participant LLM as LLM Reranker
    participant EV as Evaluator

    Q->>FE: extract(h, r, t)
    FE-->>BA: context vector x ∈ ℝ⁸
    BA-->>BA: check remaining budget
    alt budget available
        BA-->>LLM: ARM_LLM (arm=1)
        LLM-->>EV: reranked candidates
    else budget exhausted
        BA-->>EM: ARM_EMBEDDING (arm=0)
        EM-->>EV: embedding-only candidates
    end
    EV-->>BA: reward = 1 / rank
    BA->>BA: update agent, decrement budget
```

## Module Dependency Graph

```mermaid
graph TD
    config[src/config.py] --> data
    config --> models
    config --> eval
    config --> rl

    data[src/data/fb15k237.py] --> eval
    data --> rl

    models_emb[src/models/embedding_baseline.py] --> eval
    models_llm[src/models/llm_client.py] --> models_scorer
    models_scorer[src/models/scorer.py] --> models_reranker
    models_reranker[src/models/reranker.py] --> eval
    prompts[src/prompts/renderer.py] --> models_scorer

    wikidata[src/wikidata/sparql.py] --> models_scorer

    eval_cand[src/eval/candidates.py] --> eval_harness
    eval_harness[src/eval/evaluate.py] --> eval_metrics
    eval_metrics[src/eval/metrics.py]

    rl_feat[src/rl/features.py] --> rl_bandit
    rl_bandit[src/rl/bandit.py] --> rl_selector
    rl_selector[src/rl/prompt_selector.py] --> eval_harness
    rl_bandit --> rl_budget
    rl_budget[src/rl/budget_agent.py] --> rl_budget_exp
    rl_budget_exp[src/rl/budget_experiment.py]

    utils_cost[src/utils/cost_tracker.py] --> models_llm
    utils_log[src/utils/logging_config.py]
    utils_repro[src/utils/reproducibility.py] --> experiment
    experiment[src/experiment.py]
```

## Data Flow Description

| Stage | Module | Input | Output |
|---|---|---|---|
| Dataset loading | `src/data/fb15k237.py` | Raw TSV files | Train/valid/test triple lists |
| Embedding training | `src/models/embedding_baseline.py` | Training triples | RotatE model checkpoint |
| Candidate generation | `src/eval/candidates.py` | Query + embedding model | Ranked entity list (top-K) |
| Entity grounding | `src/wikidata/sparql.py` | Entity MID list | Label + description dict |
| Prompt rendering | `src/prompts/renderer.py` | Template ID + entity data | Chat message list |
| LLM scoring | `src/models/scorer.py` | Message list | `(score, reason)` per triple |
| Reranking | `src/models/reranker.py` | Candidates + scores | Sorted candidates |
| Metric computation | `src/eval/metrics.py` | `RankingResult` list | MRR, Hits@1/3/10 |
| RL feature extraction | `src/rl/features.py` | Query triple | 8-dim feature vector |
| Prompt selection | `src/rl/prompt_selector.py` | Feature vector | Template ID |
| Budget decision | `src/rl/budget_agent.py` | Feature vector + budget | ARM_EMBEDDING or ARM_LLM |
| Cost tracking | `src/utils/cost_tracker.py` | API usage dict | Per-call cost records |
