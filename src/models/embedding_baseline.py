"""PyKEEN RotatE embedding baseline for knowledge graph link prediction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from pykeen.models import RotatE
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from src.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingBaseline:
    """PyKEEN RotatE embedding baseline for link prediction."""

    def __init__(
        self,
        train_triples: list[tuple[str, str, str]],
        valid_triples: list[tuple[str, str, str]] | None = None,
        test_triples: list[tuple[str, str, str]] | None = None,
        model_name: str = "RotatE",
        embedding_dim: int = 256,
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_seed: int | None = None,
        device: str | None = None,
    ):
        settings = get_settings()
        self.random_seed = random_seed or settings.random_seed
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Convert triples to numpy arrays for PyKEEN
        self.train_triples_raw = train_triples
        self.valid_triples_raw = valid_triples or []

        # Build TriplesFactory
        self.training_factory = self._build_factory(train_triples)

        # PyKEEN pipeline requires a testing factory; fall back to a small
        # slice of training triples when none is explicitly provided.
        _test = test_triples or train_triples[:max(1, len(train_triples) // 10)]
        self.testing_factory = self._build_factory(
            _test,
            entity_to_id=self.training_factory.entity_to_id,
            relation_to_id=self.training_factory.relation_to_id,
        )

        self.validation_factory = (
            self._build_factory(
                valid_triples,
                entity_to_id=self.training_factory.entity_to_id,
                relation_to_id=self.training_factory.relation_to_id,
            )
            if valid_triples
            else None
        )

        self.model: RotatE | None = None
        self.pipeline_result = None

    def _build_factory(
        self,
        triples: list[tuple[str, str, str]],
        entity_to_id: dict[str, int] | None = None,
        relation_to_id: dict[str, int] | None = None,
    ) -> TriplesFactory:
        """Convert list of (h, r, t) to PyKEEN TriplesFactory."""
        arr = np.array(triples, dtype=str)
        if entity_to_id is not None and relation_to_id is not None:
            return TriplesFactory.from_labeled_triples(
                arr,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
        return TriplesFactory.from_labeled_triples(arr)

    def train(self) -> dict:
        """Train the RotatE model using PyKEEN pipeline."""
        logger.info(
            f"Training {self.model_name} (dim={self.embedding_dim}, "
            f"epochs={self.num_epochs}, device={self.device})"
        )

        pipeline_kwargs = dict(
            model=self.model_name,
            training=self.training_factory,
            testing=self.testing_factory,
            model_kwargs={"embedding_dim": self.embedding_dim},
            training_kwargs={
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
            },
            optimizer_kwargs={"lr": self.learning_rate},
            random_seed=self.random_seed,
            device=self.device,
        )
        if self.validation_factory:
            pipeline_kwargs["validation"] = self.validation_factory

        self.pipeline_result = pipeline(**pipeline_kwargs)
        self.model = self.pipeline_result.model

        logger.info("Training complete")
        return self._training_summary()

    def _training_summary(self) -> dict:
        return {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_epochs": self.num_epochs,
            "num_entities": self.training_factory.num_entities,
            "num_relations": self.training_factory.num_relations,
            "num_train_triples": self.training_factory.num_triples,
            "device": self.device,
        }

    def score_triple(self, h: str, r: str, t: str) -> float:
        """Score a single triple using the trained model. Higher = more plausible."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        e2id = self.training_factory.entity_to_id
        r2id = self.training_factory.relation_to_id

        # Return very low score for unknown entities/relations
        if h not in e2id or t not in e2id or r not in r2id:
            return -1e6

        h_id = torch.tensor([e2id[h]], dtype=torch.long, device=self.device)
        r_id = torch.tensor([r2id[r]], dtype=torch.long, device=self.device)
        t_id = torch.tensor([e2id[t]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            score = self.model.score_hrt(
                torch.stack([h_id, r_id, t_id], dim=1)
            )
        return float(score.item())

    def get_score_fn(self) -> Callable[[str, str, str], float]:
        """Return a scoring function compatible with the eval harness."""

        def score_fn(h: str, r: str, t: str) -> float:
            return self.score_triple(h, r, t)

        return score_fn

    def save_model(self, path: Path | None = None):
        """Save the trained model."""
        if self.pipeline_result is None:
            raise RuntimeError("No model to save. Train first.")
        settings = get_settings()
        path = path or Path(settings.results_dir) / "embedding_model"
        path.mkdir(parents=True, exist_ok=True)
        self.pipeline_result.save_to_directory(str(path))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load a previously saved model."""
        path = Path(path)
        self.model = torch.load(path / "trained_model.pkl", weights_only=False)
        self.training_factory = TriplesFactory.from_path_binary(
            str(path / "training_triples")
        )
        logger.info(f"Model loaded from {path}")
