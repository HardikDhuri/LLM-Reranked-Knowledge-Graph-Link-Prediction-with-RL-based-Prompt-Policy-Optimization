"""Tests for src/data/fb15k237.py — no network access required."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.data.fb15k237 import FB15k237Dataset, load_triples

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_TRIPLES = [
    ("/m/head1", "/relation/r1", "/m/tail1"),
    ("/m/head2", "/relation/r2", "/m/tail2"),
    ("/m/head1", "/relation/r2", "/m/tail3"),
]


@pytest.fixture()
def triple_file(tmp_path: Path) -> Path:
    """Write a small tab-separated triples file to a temp directory."""
    content = textwrap.dedent(
        """\
        /m/head1\t/relation/r1\t/m/tail1
        /m/head2\t/relation/r2\t/m/tail2
        /m/head1\t/relation/r2\t/m/tail3
        """
    )
    filepath = tmp_path / "triples.txt"
    filepath.write_text(content, encoding="utf-8")
    return filepath


@pytest.fixture()
def dataset() -> FB15k237Dataset:
    """Return a small synthetic FB15k237Dataset."""
    train = _SAMPLE_TRIPLES[:2]
    valid = [("/m/head3", "/relation/r1", "/m/tail4")]
    test = [("/m/head4", "/relation/r3", "/m/tail5")]
    return FB15k237Dataset(train=train, valid=valid, test=test)


# ---------------------------------------------------------------------------
# load_triples
# ---------------------------------------------------------------------------


def test_load_triples_returns_correct_tuples(triple_file: Path) -> None:
    triples = load_triples(triple_file)
    assert triples == _SAMPLE_TRIPLES


def test_load_triples_count(triple_file: Path) -> None:
    triples = load_triples(triple_file)
    assert len(triples) == 3


def test_load_triples_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert load_triples(empty) == []


def test_load_triples_bad_line_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("only_one_field\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 3 tab-separated fields"):
        load_triples(bad)


# ---------------------------------------------------------------------------
# FB15k237Dataset properties
# ---------------------------------------------------------------------------


def test_all_triples(dataset: FB15k237Dataset) -> None:
    assert dataset.all_triples == dataset.train + dataset.valid + dataset.test


def test_all_triples_set_size(dataset: FB15k237Dataset) -> None:
    assert len(dataset.all_triples_set) == len(dataset.all_triples)
    assert isinstance(dataset.all_triples_set, set)


def test_entities(dataset: FB15k237Dataset) -> None:
    # Heads and tails from all splits.
    expected = sorted(
        {
            "/m/head1",
            "/m/tail1",
            "/m/head2",
            "/m/tail2",
            "/m/head3",
            "/m/tail4",
            "/m/head4",
            "/m/tail5",
        }
    )
    assert dataset.entities == expected


def test_relations(dataset: FB15k237Dataset) -> None:
    expected = sorted({"/relation/r1", "/relation/r2", "/relation/r3"})
    assert dataset.relations == expected


def test_summary_keys(dataset: FB15k237Dataset) -> None:
    s = dataset.summary()
    assert set(s.keys()) == {"train", "valid", "test", "all", "entities", "relations"}


def test_summary_values(dataset: FB15k237Dataset) -> None:
    s = dataset.summary()
    assert s["train"] == 2
    assert s["valid"] == 1
    assert s["test"] == 1
    assert s["all"] == 4
    assert s["entities"] == 8
    assert s["relations"] == 3


# ---------------------------------------------------------------------------
# all_triples_set deduplication
# ---------------------------------------------------------------------------


def test_all_triples_set_deduplicates() -> None:
    triple = ("/m/h", "/r", "/m/t")
    ds = FB15k237Dataset(train=[triple], valid=[triple], test=[])
    assert len(ds.all_triples) == 2  # list keeps duplicates
    assert len(ds.all_triples_set) == 1  # set deduplicates
