"""Tests for quant_tools.predict.base — PredictionModel ABC and ModelState."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import ModelState, PredictionModel


# -- Concrete stub for testing the ABC --


class _StubModel(PredictionModel):
    """Minimal concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "stub_model"

    def train(self, features_df, target) -> None:
        self.state.is_trained = True
        self.state.training_samples = len(target)
        self.state.last_trained = datetime.now()

    def predict(self, features_df, horizon_days: int = 5) -> PredictionResult:
        return PredictionResult(
            symbol="TEST",
            asset_class=AssetClass.EQUITY,
            model_name=self.name,
            direction=PredictionDirection.UP,
            confidence=0.75,
            horizon_days=horizon_days,
        )


# ── ModelState tests ──


class TestModelState:
    def test_defaults(self):
        state = ModelState()
        assert state.is_trained is False
        assert state.training_samples == 0
        assert state.last_trained is None
        assert state.validation_score == 0.0
        assert state.metadata == {}

    def test_metadata_independent(self):
        """Each ModelState gets its own dict (no shared mutable default)."""
        a = ModelState()
        b = ModelState()
        a.metadata["key"] = "value"
        assert b.metadata == {}

    def test_set_trained(self):
        state = ModelState()
        state.is_trained = True
        state.training_samples = 100
        state.last_trained = datetime(2025, 1, 1)
        state.validation_score = 0.85
        assert state.is_trained is True
        assert state.training_samples == 100
        assert state.validation_score == 0.85


# ── PredictionModel ABC tests ──


class TestPredictionModelABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            PredictionModel()

    def test_stub_instantiates(self):
        model = _StubModel()
        assert model.name == "stub_model"
        assert model.state.is_trained is False

    def test_train_updates_state(self):
        model = _StubModel()
        features = pd.DataFrame({"a": [1, 2, 3]})
        target = np.array([0.01, -0.02, 0.03])
        model.train(features, target)
        assert model.state.is_trained is True
        assert model.state.training_samples == 3
        assert model.state.last_trained is not None

    def test_predict_returns_prediction_result(self):
        model = _StubModel()
        features = pd.DataFrame({"a": [1]})
        result = model.predict(features, horizon_days=10)
        assert isinstance(result, PredictionResult)
        assert result.symbol == "TEST"
        assert result.horizon_days == 10
        assert result.direction == PredictionDirection.UP

    def test_predict_default_horizon(self):
        model = _StubModel()
        result = model.predict(pd.DataFrame({"a": [1]}))
        assert result.horizon_days == 5

    def test_supports_asset_class_default_all(self):
        model = _StubModel()
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True

    def test_name_is_property(self):
        model = _StubModel()
        assert isinstance(type(model).name, property)

    def test_state_is_model_state_instance(self):
        model = _StubModel()
        assert isinstance(model.state, ModelState)

    def test_multiple_models_independent_state(self):
        a = _StubModel()
        b = _StubModel()
        a.train(pd.DataFrame({"x": [1]}), np.array([0.1]))
        assert a.state.is_trained is True
        assert b.state.is_trained is False

    def test_predict_confidence(self):
        model = _StubModel()
        result = model.predict(pd.DataFrame({"a": [1]}))
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_model_name_matches(self):
        model = _StubModel()
        result = model.predict(pd.DataFrame({"a": [1]}))
        assert result.model_name == model.name


# ── Custom override test ──


class _EquityOnlyModel(PredictionModel):
    """Model that only supports equities."""

    @property
    def name(self) -> str:
        return "equity_only"

    def train(self, features_df, target) -> None:
        pass

    def predict(self, features_df, horizon_days: int = 5) -> PredictionResult:
        return PredictionResult(
            symbol="X",
            asset_class=AssetClass.EQUITY,
            model_name=self.name,
            direction=PredictionDirection.NEUTRAL,
        )

    def supports_asset_class(self, asset_class: AssetClass) -> bool:
        return asset_class == AssetClass.EQUITY


class TestCustomAssetClassSupport:
    def test_equity_supported(self):
        model = _EquityOnlyModel()
        assert model.supports_asset_class(AssetClass.EQUITY) is True

    def test_crypto_not_supported(self):
        model = _EquityOnlyModel()
        assert model.supports_asset_class(AssetClass.CRYPTO) is False

    def test_forex_not_supported(self):
        model = _EquityOnlyModel()
        assert model.supports_asset_class(AssetClass.FOREX) is False
