"""Base prediction model ABC and shared state."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime

from quant_tools.models import AssetClass, PredictionResult


@dataclass
class ModelState:
    """Tracks model training state."""

    is_trained: bool = False
    last_trained: datetime | None = None
    training_samples: int = 0
    validation_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class PredictionModel(abc.ABC):
    """Abstract base class for all prediction models."""

    def __init__(self) -> None:
        self.state = ModelState()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique model identifier."""
        ...

    @abc.abstractmethod
    def train(self, features_df: object, target: object) -> None:
        """Train the model on feature data."""
        ...

    @abc.abstractmethod
    def predict(self, features_df: object, horizon_days: int = 5, interval: str = "1d") -> PredictionResult:
        """Generate a prediction."""
        ...

    def supports_asset_class(self, asset_class: AssetClass) -> bool:
        """Whether this model supports the given asset class."""
        return True
