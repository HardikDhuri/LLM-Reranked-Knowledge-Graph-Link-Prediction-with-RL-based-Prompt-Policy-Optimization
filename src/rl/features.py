"""Query feature extraction for contextual bandit prompt selection."""

from __future__ import annotations

import numpy as np


class QueryFeatureExtractor:
    """Extracts contextual features from a (h, r, t) query for the bandit."""

    def __init__(
        self,
        entity_frequency: dict[str, int] | None = None,
        relation_frequency: dict[str, int] | None = None,
        num_entities: int = 1,
        num_relations: int = 1,
    ):
        self.entity_freq = entity_frequency or {}
        self.relation_freq = relation_frequency or {}
        self.num_entities = num_entities
        self.num_relations = num_relations

    @classmethod
    def from_triples(cls, triples: list[tuple[str, str, str]]) -> QueryFeatureExtractor:
        """Build feature extractor from training triples."""
        entity_freq: dict[str, int] = {}
        relation_freq: dict[str, int] = {}
        for h, r, t in triples:
            entity_freq[h] = entity_freq.get(h, 0) + 1
            entity_freq[t] = entity_freq.get(t, 0) + 1
            relation_freq[r] = relation_freq.get(r, 0) + 1
        entities = set()
        relations = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        return cls(
            entity_frequency=entity_freq,
            relation_frequency=relation_freq,
            num_entities=len(entities),
            num_relations=len(relations),
        )

    def extract(self, h: str, r: str, t: str) -> np.ndarray:
        """
        Extract a feature vector for a query triple.

        Features:
          0: head entity frequency (normalized)
          1: tail entity frequency (normalized)
          2: relation frequency (normalized)
          3: head frequency rank (normalized)
          4: tail frequency rank (normalized)
          5: relation frequency rank (normalized)
          6: is head frequent (> median freq)
          7: is tail frequent (> median freq)
        """
        max_ent_freq = max(self.entity_freq.values()) if self.entity_freq else 1
        max_rel_freq = max(self.relation_freq.values()) if self.relation_freq else 1

        h_freq = self.entity_freq.get(h, 0) / max_ent_freq
        t_freq = self.entity_freq.get(t, 0) / max_ent_freq
        r_freq = self.relation_freq.get(r, 0) / max_rel_freq

        # Frequency ranks
        sorted_ents = sorted(self.entity_freq.values(), reverse=True)
        sorted_rels = sorted(self.relation_freq.values(), reverse=True)
        h_raw = self.entity_freq.get(h, 0)
        t_raw = self.entity_freq.get(t, 0)
        r_raw = self.relation_freq.get(r, 0)

        n_ents = max(len(sorted_ents), 1)
        n_rels = max(len(sorted_rels), 1)
        h_rank = sorted_ents.index(h_raw) / n_ents if h_raw in sorted_ents else 1.0
        t_rank = sorted_ents.index(t_raw) / n_ents if t_raw in sorted_ents else 1.0
        r_rank = sorted_rels.index(r_raw) / n_rels if r_raw in sorted_rels else 1.0

        median_ent_freq = (
            np.median(list(self.entity_freq.values())) if self.entity_freq else 0
        )
        h_is_frequent = float(h_raw > median_ent_freq)
        t_is_frequent = float(t_raw > median_ent_freq)

        return np.array(
            [
                h_freq,
                t_freq,
                r_freq,
                h_rank,
                t_rank,
                r_rank,
                h_is_frequent,
                t_is_frequent,
            ],
            dtype=np.float32,
        )

    @property
    def feature_dim(self) -> int:
        return 8
