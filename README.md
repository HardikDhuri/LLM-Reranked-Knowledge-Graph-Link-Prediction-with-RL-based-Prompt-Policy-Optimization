# KG-LLM-RL Link Prediction

RL-Optimized LLM Reranking for Knowledge Graph Link Prediction: generate top-K candidates with KG embeddings, ground entities via Wikidata SPARQL (labels/descriptions), rerank triples using an LLM, and train an RL policy to select prompts/budgets for best MRR/Hits@K under cost constraints.

## Setup

1. Install the package and dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Copy the example environment file and fill in your credentials:

   ```bash
   cp .env.example .env
   # edit .env with your API key and settings
   ```

## Run Tests

```bash
pytest tests/ -v
```
