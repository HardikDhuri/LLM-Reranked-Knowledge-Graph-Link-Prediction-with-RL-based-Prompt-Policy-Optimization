"""Candidate generation and filtering for KG link prediction evaluation."""

from __future__ import annotations

import random


def _make_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def generate_tail_candidates(
    h: str,
    r: str,
    true_t: str,
    all_entities: list[str],
    num_candidates: int,
    seed: int | None = None,
) -> list[str]:
    """Generate candidate tails: true tail + random negatives."""
    rng = _make_rng(seed)
    cands: set[str] = {true_t}
    while len(cands) < min(num_candidates, len(all_entities)):
        cands.add(rng.choice(all_entities))
    return list(cands)


def generate_head_candidates(
    true_h: str,
    r: str,
    t: str,
    all_entities: list[str],
    num_candidates: int,
    seed: int | None = None,
) -> list[str]:
    """Generate candidate heads: true head + random negatives."""
    rng = _make_rng(seed)
    cands: set[str] = {true_h}
    while len(cands) < min(num_candidates, len(all_entities)):
        cands.add(rng.choice(all_entities))
    return list(cands)


def filter_candidates_tail(
    h: str,
    r: str,
    candidates: list[str],
    true_t: str,
    known_triples: set[tuple[str, str, str]],
) -> list[str]:
    """
    Filtered setting: remove candidates that form a known-true triple
    (except the target true tail).
    This is the standard KG link prediction evaluation protocol.
    """
    filtered = []
    for t in candidates:
        if t == true_t:
            filtered.append(t)
        elif (h, r, t) not in known_triples:
            filtered.append(t)
    if true_t not in filtered:
        filtered.append(true_t)
    return filtered


def filter_candidates_head(
    candidates: list[str],
    r: str,
    t: str,
    true_h: str,
    known_triples: set[tuple[str, str, str]],
) -> list[str]:
    """Filtered setting for head prediction."""
    filtered = []
    for h in candidates:
        if h == true_h:
            filtered.append(h)
        elif (h, r, t) not in known_triples:
            filtered.append(h)
    if true_h not in filtered:
        filtered.append(true_h)
    return filtered
