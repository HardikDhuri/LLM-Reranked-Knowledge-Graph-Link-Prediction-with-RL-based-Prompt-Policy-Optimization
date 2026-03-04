"""Unit tests for src/models/embedding_baseline.py.

Uses a tiny synthetic KG — no real dataset download required.
"""

from __future__ import annotations

import pytest

from src.models.embedding_baseline import EmbeddingBaseline

# ---------------------------------------------------------------------------
# Tiny synthetic KG
# ---------------------------------------------------------------------------

ENTITIES = ["e0", "e1", "e2", "e3", "e4"]
RELATIONS = ["r0", "r1"]

# 20 synthetic triples (with repetition removed implicitly by list)
TRAIN_TRIPLES: list[tuple[str, str, str]] = [
    ("e0", "r0", "e1"),
    ("e0", "r0", "e2"),
    ("e0", "r1", "e3"),
    ("e1", "r0", "e2"),
    ("e1", "r0", "e3"),
    ("e1", "r1", "e4"),
    ("e2", "r0", "e3"),
    ("e2", "r0", "e4"),
    ("e2", "r1", "e0"),
    ("e3", "r0", "e4"),
    ("e3", "r0", "e0"),
    ("e3", "r1", "e1"),
    ("e4", "r0", "e0"),
    ("e4", "r0", "e1"),
    ("e4", "r1", "e2"),
    ("e0", "r1", "e4"),
    ("e1", "r1", "e3"),
    ("e2", "r1", "e4"),
    ("e3", "r1", "e0"),
    ("e4", "r1", "e3"),
]

VALID_TRIPLES: list[tuple[str, str, str]] = [
    ("e0", "r0", "e3"),
    ("e1", "r1", "e0"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline(**kwargs) -> EmbeddingBaseline:
    defaults = dict(
        train_triples=TRAIN_TRIPLES,
        embedding_dim=16,
        num_epochs=1,
        batch_size=16,
        random_seed=42,
        device="cpu",
    )
    defaults.update(kwargs)
    return EmbeddingBaseline(**defaults)


def _trained_baseline(**kwargs) -> EmbeddingBaseline:
    bl = _make_baseline(**kwargs)
    bl.train()
    return bl


# ---------------------------------------------------------------------------
# __init__ / TriplesFactory creation
# ---------------------------------------------------------------------------


def test_init_creates_training_factory():
    bl = _make_baseline()
    assert bl.training_factory is not None


def test_init_entity_count():
    bl = _make_baseline()
    assert bl.training_factory.num_entities == len(ENTITIES)


def test_init_relation_count():
    bl = _make_baseline()
    assert bl.training_factory.num_relations == len(RELATIONS)


def test_init_triple_count():
    bl = _make_baseline()
    assert bl.training_factory.num_triples == len(TRAIN_TRIPLES)


def test_init_no_validation_factory_by_default():
    bl = _make_baseline()
    assert bl.validation_factory is None


def test_init_validation_factory_created_when_provided():
    bl = _make_baseline(valid_triples=VALID_TRIPLES)
    assert bl.validation_factory is not None


def test_init_model_is_none_before_training():
    bl = _make_baseline()
    assert bl.model is None


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_runs_without_error():
    bl = _make_baseline()
    bl.train()  # should not raise


def test_train_returns_summary_dict():
    bl = _make_baseline()
    summary = bl.train()
    assert isinstance(summary, dict)


def test_train_summary_keys():
    bl = _make_baseline()
    summary = bl.train()
    expected_keys = {
        "model",
        "embedding_dim",
        "num_epochs",
        "num_entities",
        "num_relations",
        "num_train_triples",
        "device",
    }
    assert expected_keys.issubset(summary.keys())


def test_train_sets_model():
    bl = _make_baseline()
    bl.train()
    assert bl.model is not None


def test_train_with_validation():
    bl = _make_baseline(valid_triples=VALID_TRIPLES)
    bl.train()  # should not raise
    assert bl.model is not None


# ---------------------------------------------------------------------------
# score_triple()
# ---------------------------------------------------------------------------


def test_score_triple_raises_if_not_trained():
    bl = _make_baseline()
    with pytest.raises(RuntimeError, match="[Tt]rain"):
        bl.score_triple("e0", "r0", "e1")


def test_score_triple_returns_float_for_known():
    bl = _trained_baseline()
    score = bl.score_triple("e0", "r0", "e1")
    assert isinstance(score, float)


def test_score_triple_returns_minus_1e6_for_unknown_head():
    bl = _trained_baseline()
    score = bl.score_triple("UNKNOWN", "r0", "e1")
    assert score == -1e6


def test_score_triple_returns_minus_1e6_for_unknown_tail():
    bl = _trained_baseline()
    score = bl.score_triple("e0", "r0", "UNKNOWN")
    assert score == -1e6


def test_score_triple_returns_minus_1e6_for_unknown_relation():
    bl = _trained_baseline()
    score = bl.score_triple("e0", "UNKNOWN_REL", "e1")
    assert score == -1e6


# ---------------------------------------------------------------------------
# get_score_fn()
# ---------------------------------------------------------------------------


def test_get_score_fn_returns_callable():
    bl = _trained_baseline()
    fn = bl.get_score_fn()
    assert callable(fn)


def test_get_score_fn_callable_returns_float():
    bl = _trained_baseline()
    fn = bl.get_score_fn()
    result = fn("e0", "r0", "e1")
    assert isinstance(result, float)


def test_get_score_fn_unknown_returns_minus_1e6():
    bl = _trained_baseline()
    fn = bl.get_score_fn()
    assert fn("UNKNOWN", "r0", "e1") == -1e6


# ---------------------------------------------------------------------------
# save_model() / load_model()
# ---------------------------------------------------------------------------


def test_save_model_raises_if_not_trained(tmp_path):
    bl = _make_baseline()
    with pytest.raises(RuntimeError, match="[Tt]rain"):
        bl.save_model(tmp_path / "model")


def test_save_and_load_model_roundtrip(tmp_path):
    bl = _trained_baseline()
    save_path = tmp_path / "model"
    bl.save_model(save_path)

    # Verify files were written
    assert save_path.exists()

    # Load into a fresh baseline (we only need the model attribute)
    bl2 = _make_baseline()
    bl2.load_model(save_path)
    assert bl2.model is not None


def test_loaded_model_can_score(tmp_path):
    bl = _trained_baseline()
    save_path = tmp_path / "model"
    bl.save_model(save_path)

    bl2 = _make_baseline()
    bl2.load_model(save_path)
    bl2.model.to("cpu")
    # Score via the loaded model directly
    import torch

    e2id = bl.training_factory.entity_to_id
    r2id = bl.training_factory.relation_to_id
    h_id = torch.tensor([e2id["e0"]], dtype=torch.long)
    r_id = torch.tensor([r2id["r0"]], dtype=torch.long)
    t_id = torch.tensor([e2id["e1"]], dtype=torch.long)
    with torch.no_grad():
        score = bl2.model.score_hrt(torch.stack([h_id, r_id, t_id], dim=1))
    assert isinstance(float(score.item()), float)
