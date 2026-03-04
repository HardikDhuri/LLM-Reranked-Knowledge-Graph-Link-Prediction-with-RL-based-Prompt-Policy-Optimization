"""Tests for src/eval/candidates.py."""

from __future__ import annotations

from src.eval.candidates import (
    filter_candidates_head,
    filter_candidates_tail,
    generate_head_candidates,
    generate_tail_candidates,
)

ENTITIES = [f"e{i}" for i in range(20)]


# ---------------------------------------------------------------------------
# generate_tail_candidates
# ---------------------------------------------------------------------------


def test_generate_tail_candidates_includes_true_tail() -> None:
    cands = generate_tail_candidates("h", "r", "e0", ENTITIES, num_candidates=5, seed=1)
    assert "e0" in cands


def test_generate_tail_candidates_correct_count() -> None:
    cands = generate_tail_candidates("h", "r", "e0", ENTITIES, num_candidates=8, seed=2)
    assert len(cands) == 8


def test_generate_tail_candidates_deterministic() -> None:
    cands1 = generate_tail_candidates("h", "r", "e0", ENTITIES, num_candidates=5, seed=42)
    cands2 = generate_tail_candidates("h", "r", "e0", ENTITIES, num_candidates=5, seed=42)
    assert cands1 == cands2


def test_generate_tail_candidates_capped_at_entity_count() -> None:
    small_entities = ["a", "b", "c"]
    cands = generate_tail_candidates("h", "r", "a", small_entities, num_candidates=100, seed=0)
    assert len(cands) == len(small_entities)


# ---------------------------------------------------------------------------
# filter_candidates_tail
# ---------------------------------------------------------------------------


def test_filter_tail_removes_known_triples() -> None:
    known = {("h", "r", "e1"), ("h", "r", "e2")}
    candidates = ["e0", "e1", "e2"]
    # e0 is true tail, e1 and e2 are known (should be removed)
    filtered = filter_candidates_tail("h", "r", candidates, true_t="e0", known_triples=known)
    assert "e0" in filtered
    assert "e1" not in filtered
    assert "e2" not in filtered


def test_filter_tail_keeps_true_tail_even_if_known() -> None:
    # true_t is in known_triples but must still be kept
    known = {("h", "r", "e0")}
    candidates = ["e0", "e1"]
    filtered = filter_candidates_tail("h", "r", candidates, true_t="e0", known_triples=known)
    assert "e0" in filtered


def test_filter_tail_appends_true_tail_if_missing() -> None:
    # true_t not in candidates at all — must be appended
    known: set = set()
    candidates = ["e1", "e2"]
    filtered = filter_candidates_tail("h", "r", candidates, true_t="e0", known_triples=known)
    assert "e0" in filtered


# ---------------------------------------------------------------------------
# filter_candidates_head
# ---------------------------------------------------------------------------


def test_filter_head_removes_known_triples() -> None:
    known = {("e1", "r", "t"), ("e2", "r", "t")}
    candidates = ["e0", "e1", "e2"]
    filtered = filter_candidates_head(candidates, "r", "t", true_h="e0", known_triples=known)
    assert "e0" in filtered
    assert "e1" not in filtered
    assert "e2" not in filtered


def test_filter_head_keeps_true_head_even_if_known() -> None:
    known = {("e0", "r", "t")}
    candidates = ["e0", "e1"]
    filtered = filter_candidates_head(candidates, "r", "t", true_h="e0", known_triples=known)
    assert "e0" in filtered


def test_filter_head_appends_true_head_if_missing() -> None:
    known: set = set()
    candidates = ["e1", "e2"]
    filtered = filter_candidates_head(candidates, "r", "t", true_h="e0", known_triples=known)
    assert "e0" in filtered


# ---------------------------------------------------------------------------
# generate_head_candidates
# ---------------------------------------------------------------------------


def test_generate_head_candidates_includes_true_head() -> None:
    cands = generate_head_candidates("e0", "r", "t", ENTITIES, num_candidates=5, seed=1)
    assert "e0" in cands


def test_generate_head_candidates_deterministic() -> None:
    cands1 = generate_head_candidates("e0", "r", "t", ENTITIES, num_candidates=5, seed=7)
    cands2 = generate_head_candidates("e0", "r", "t", ENTITIES, num_candidates=5, seed=7)
    assert cands1 == cands2
