"""LLM-based candidate reranker for KG link prediction."""

import logging
import time

from src.eval.metrics import RankingResult
from src.models.scorer import TripleScorer

logger = logging.getLogger(__name__)


class LLMReranker:
    """Reranks candidate entities for a query using LLM triple scoring."""

    def __init__(self, scorer: TripleScorer):
        self.scorer = scorer
        self.total_queries = 0
        self.total_rerank_time = 0.0

    def rerank_tail_candidates(
        self,
        head: str,
        relation: str,
        true_tail: str,
        candidates: list[str],
    ) -> RankingResult:
        """Rerank tail candidates for query (head, relation, ?).

        Scores each candidate triple with the LLM and returns a RankingResult.
        """
        start = time.time()
        self.total_queries += 1

        scored = []
        for tail in candidates:
            score = self.scorer.score_triple(head, relation, tail)
            scored.append((tail, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Find rank of true tail (1-based)
        ranked_entities = [t for t, _ in scored]
        if true_tail in ranked_entities:
            true_rank = ranked_entities.index(true_tail) + 1
        else:
            true_rank = len(ranked_entities) + 1
            logger.warning("True tail %s not in candidates!", true_tail)

        elapsed = time.time() - start
        self.total_rerank_time += elapsed

        logger.debug(
            "Reranked %d candidates for (%s, %s, ?) in %.2fs — true_rank=%d",
            len(candidates),
            head,
            relation,
            elapsed,
            true_rank,
        )

        return RankingResult(
            query=(head, relation, true_tail),
            true_rank=true_rank,
            num_candidates=len(candidates),
            scored_candidates=scored,
        )

    def rerank_batch(
        self,
        queries: list[tuple[str, str, str]],
        candidates_per_query: dict[tuple[str, str, str], list[str]],
        progress: bool = True,
    ) -> list[RankingResult]:
        """Rerank a batch of queries."""
        results = []
        iterator = queries
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(queries, desc="Reranking")
            except ImportError:
                pass

        for query in iterator:
            h, r, t = query
            cands = candidates_per_query.get(query, [])
            if not cands:
                logger.warning("No candidates for query %s, skipping", query)
                continue
            result = self.rerank_tail_candidates(h, r, t, cands)
            results.append(result)

        return results

    def stats(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_rerank_time_s": round(self.total_rerank_time, 2),
            "avg_time_per_query_s": round(
                self.total_rerank_time / max(1, self.total_queries), 2
            ),
            "scorer_stats": self.scorer.stats(),
        }
