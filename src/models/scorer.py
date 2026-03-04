"""LLM-based triple scorer."""

import logging

from src.models.llm_client import LLMClient
from src.prompts.renderer import PromptTemplate
from src.wikidata.sparql import WikidataResolver

logger = logging.getLogger(__name__)


class TripleScorer:
    """Scores a (head, relation, tail) triple using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        resolver: WikidataResolver,
        template: PromptTemplate,
    ):
        self.llm = llm_client
        self.resolver = resolver
        self.template = template
        # Score cache: (h, r, t, template_id) -> score
        self._score_cache: dict[tuple[str, str, str, str], float] = {}
        self.total_scored = 0
        self.total_cache_hits = 0

    def _resolve_entity(self, mid: str) -> dict[str, str]:
        """Resolve a Freebase MID to label + description."""
        return self.resolver.mid_to_text(mid)

    def score_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        use_cache: bool = True,
    ) -> float:
        """Score a single triple (h, r, t).

        Returns a float score in [0, 1]. Returns 0.0 on failure.
        """
        cache_key = (head, relation, tail, self.template.id)
        if use_cache and cache_key in self._score_cache:
            self.total_cache_hits += 1
            return self._score_cache[cache_key]

        # Resolve entities
        head_info = self._resolve_entity(head)
        tail_info = self._resolve_entity(tail)

        # Clean up relation (e.g., /people/person/nationality -> person nationality)
        relation_clean = self._clean_relation(relation)

        # Render prompt
        messages = self.template.render(
            head_label=head_info["label"],
            relation=relation_clean,
            tail_label=tail_info["label"],
            head_description=head_info.get("description", ""),
            tail_description=tail_info.get("description", ""),
        )

        # Call LLM
        try:
            result = self.llm.chat_completion_json(messages=messages)
            parsed = result.get("parsed_json")
            if parsed:
                # Handle different template output formats
                score = self._extract_score(parsed)
            else:
                logger.warning("No parsed JSON for (%s, %s, %s)", head, relation, tail)
                score = 0.0
        except Exception as e:
            logger.error(
                "LLM scoring failed for (%s, %s, %s): %s", head, relation, tail, e
            )
            score = 0.0

        self._score_cache[cache_key] = score
        self.total_scored += 1
        return score

    @staticmethod
    def _extract_score(parsed: dict) -> float:
        """Extract a float score from parsed JSON. Handle different formats."""
        # Direct score field
        if "score" in parsed:
            try:
                return float(parsed["score"])
            except (ValueError, TypeError):
                logger.warning(
                    "Could not convert 'score' field to float: %r", parsed["score"]
                )
        # Binary judge format: convert confidence to score
        if "judgment" in parsed and "confidence" in parsed:
            try:
                conf = float(parsed["confidence"])
                return conf if parsed["judgment"].lower() == "true" else 1.0 - conf
            except (ValueError, TypeError):
                pass
        # Fallback
        return 0.0

    @staticmethod
    def _clean_relation(relation: str) -> str:
        """Convert Freebase relation path to readable text.

        e.g., /people/person/nationality -> person nationality
        """
        parts = relation.strip("/").split("/")
        # Take last 2 meaningful parts
        if len(parts) >= 2:
            return " ".join(parts[-2:]).replace("_", " ")
        return relation.replace("/", " ").replace("_", " ").strip()

    def stats(self) -> dict:
        return {
            "total_scored": self.total_scored,
            "cache_hits": self.total_cache_hits,
            "template_id": self.template.id,
            "llm_stats": self.llm.stats(),
        }

    def clear_cache(self):
        self._score_cache.clear()
