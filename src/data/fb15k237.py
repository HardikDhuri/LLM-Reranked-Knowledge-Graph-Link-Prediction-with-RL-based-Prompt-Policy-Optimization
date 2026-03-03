"""FB15k-237 dataset download, extraction, and loading utilities."""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.config import get_settings

FB15K237_URL = "https://data.dgl.ai/dataset/FB15k-237.tgz"

# The archive extracts to a sub-directory with this name.
_ARCHIVE_SUBDIR = "FB15k-237"


# ---------------------------------------------------------------------------
# Download & extraction
# ---------------------------------------------------------------------------


def _reporthook(t: tqdm) -> object:
    """Return a tqdm-compatible reporthook for urllib.request.urlretrieve."""
    last_b = [0]

    def inner(b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_fb15k237(data_dir: Path) -> Path:
    """Download and extract FB15k-237 into *data_dir*.

    Returns the path to the folder that contains ``train.txt``,
    ``valid.txt``, and ``test.txt``.  Skips the download if the
    archive already exists.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / "FB15k-237.tgz"
    dataset_dir = data_dir / _ARCHIVE_SUBDIR

    if not archive_path.exists():
        print(f"Downloading FB15k-237 from {FB15K237_URL} …")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
            urllib.request.urlretrieve(FB15K237_URL, archive_path, _reporthook(t))
        print(f"Saved to {archive_path}")

    if not dataset_dir.exists():
        print(f"Extracting {archive_path} …")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
        print(f"Extracted to {dataset_dir}")

    return dataset_dir


# ---------------------------------------------------------------------------
# Triple loading
# ---------------------------------------------------------------------------


def load_triples(filepath: Path) -> list[tuple[str, str, str]]:
    """Read tab-separated triples from *filepath*.

    Each line must be ``head\\trelation\\ttail``.
    Returns a list of ``(h, r, t)`` string tuples.
    """
    triples: list[tuple[str, str, str]] = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 tab-separated fields, got {len(parts)}: {line!r}"
                )
            h, r, t = parts
            triples.append((h, r, t))
    return triples


# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------


@dataclass
class FB15k237Dataset:
    """Container for the three FB15k-237 splits."""

    train: list[tuple[str, str, str]]
    valid: list[tuple[str, str, str]]
    test: list[tuple[str, str, str]]

    @property
    def all_triples(self) -> list[tuple[str, str, str]]:
        return self.train + self.valid + self.test

    @property
    def all_triples_set(self) -> set[tuple[str, str, str]]:
        return set(self.all_triples)

    @property
    def entities(self) -> list[str]:
        ents: set[str] = set()
        for h, _r, t in self.all_triples:
            ents.add(h)
            ents.add(t)
        return sorted(ents)

    @property
    def relations(self) -> list[str]:
        return sorted({r for _, r, _ in self.all_triples})

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
            "all": len(self.all_triples),
            "entities": len(self.entities),
            "relations": len(self.relations),
        }


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def load_fb15k237(data_dir: Path | None = None) -> FB15k237Dataset:
    """Download (if necessary) and load the FB15k-237 dataset.

    Parameters
    ----------
    data_dir:
        Directory used for storing the downloaded archive and extracted
        files.  Defaults to ``get_settings().data_dir``.

    Returns
    -------
    FB15k237Dataset
        Dataclass containing train, valid, and test triple lists.
    """
    if data_dir is None:
        data_dir = get_settings().data_dir

    dataset_dir = download_fb15k237(data_dir)

    train = load_triples(dataset_dir / "train.txt")
    valid = load_triples(dataset_dir / "valid.txt")
    test = load_triples(dataset_dir / "test.txt")

    return FB15k237Dataset(train=train, valid=valid, test=test)
