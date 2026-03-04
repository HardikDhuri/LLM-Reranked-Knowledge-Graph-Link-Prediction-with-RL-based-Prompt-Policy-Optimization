"""Tests for src/rl/features.py — QueryFeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from src.rl.features import QueryFeatureExtractor

# ---------------------------------------------------------------------------
# Sample triples for testing
# ---------------------------------------------------------------------------

TRIPLES = [
    ("alice", "knows", "bob"),
    ("alice", "knows", "carol"),
    ("bob", "knows", "carol"),
    ("carol", "likes", "alice"),
    ("bob", "likes", "carol"),
]


# ---------------------------------------------------------------------------
# from_triples
# ---------------------------------------------------------------------------


def test_from_triples_entity_frequency():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    # alice appears as head in 2 triples and tail in 1 → freq 3
    assert extractor.entity_freq["alice"] == 3
    # bob appears as head in 2 triples and tail in 1 → freq 3
    assert extractor.entity_freq["bob"] == 3
    # carol appears as head in 2 triples and tail in 2 → freq 4
    assert extractor.entity_freq["carol"] == 4


def test_from_triples_relation_frequency():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    assert extractor.relation_freq["knows"] == 3
    assert extractor.relation_freq["likes"] == 2


def test_from_triples_num_entities():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    assert extractor.num_entities == 3  # alice, bob, carol


def test_from_triples_num_relations():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    assert extractor.num_relations == 2  # knows, likes


def test_from_triples_empty():
    extractor = QueryFeatureExtractor.from_triples([])
    assert extractor.entity_freq == {}
    assert extractor.relation_freq == {}
    assert extractor.num_entities == 0
    assert extractor.num_relations == 0


# ---------------------------------------------------------------------------
# extract — shape and dtype
# ---------------------------------------------------------------------------


def test_extract_returns_numpy_array():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    assert isinstance(feat, np.ndarray)


def test_extract_correct_shape():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    assert feat.shape == (extractor.feature_dim,)
    assert feat.shape == (8,)


def test_extract_dtype_float32():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    assert feat.dtype == np.float32


# ---------------------------------------------------------------------------
# extract — value ranges
# ---------------------------------------------------------------------------


def test_extract_normalized_features_in_0_1():
    """Normalized frequency features (indices 0-2) must be in [0, 1]."""
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    for i in range(3):
        assert 0.0 <= feat[i] <= 1.0, f"Feature {i} out of range: {feat[i]}"


def test_extract_rank_features_in_0_1():
    """Rank features (indices 3-5) must be in [0, 1]."""
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    for i in range(3, 6):
        assert 0.0 <= feat[i] <= 1.0, f"Feature {i} out of range: {feat[i]}"


def test_extract_binary_features_are_0_or_1():
    """Binary features (indices 6-7) must be exactly 0.0 or 1.0."""
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    assert feat[6] in (0.0, 1.0)
    assert feat[7] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# extract — known vs unknown entities
# ---------------------------------------------------------------------------


def test_extract_known_entity_nonzero_freq():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "bob")
    # alice and bob are known → h_freq and t_freq should be > 0
    assert feat[0] > 0.0  # h_freq
    assert feat[1] > 0.0  # t_freq


def test_extract_unknown_entity_zero_freq():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("unknown_head", "knows", "unknown_tail")
    assert feat[0] == pytest.approx(0.0)  # h_freq
    assert feat[1] == pytest.approx(0.0)  # t_freq


def test_extract_unknown_relation_zero_freq():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "unknown_rel", "bob")
    assert feat[2] == pytest.approx(0.0)  # r_freq


# ---------------------------------------------------------------------------
# feature_dim property
# ---------------------------------------------------------------------------


def test_feature_dim_is_8():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    assert extractor.feature_dim == 8


def test_feature_dim_matches_extract_output():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat = extractor.extract("alice", "knows", "carol")
    assert len(feat) == extractor.feature_dim


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_extract_same_query_returns_same_features():
    extractor = QueryFeatureExtractor.from_triples(TRIPLES)
    feat1 = extractor.extract("alice", "knows", "bob")
    feat2 = extractor.extract("alice", "knows", "bob")
    assert np.array_equal(feat1, feat2)
