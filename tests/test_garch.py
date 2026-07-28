"""Tests for GARCHModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.garch import GARCHModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return GARCHModel()


@pytest.fixture()
def price_df():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.DataFrame({"close": prices})


@pytest.fixture()
def trained_model(price_df):
    m = GARCHModel()
    m.train(price_df, target=None)
    return m


# ── Basic tests ─────────────────────────────────────────────────────


class TestGARCHBasic:
    def test_name(self, model):
        assert model.name == "garch"

    def test_initial_state(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True

    def test_predict_without_training(self, model):
        result = model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3


# ── Training tests ──────────────────────────────────────────────────


class TestGARCHTraining:
    def test_train_success(self, model, price_df):
        model.train(price_df, target=None)
        assert model.state.is_trained is True
        assert model.state.training_samples == 199  # diff reduces by 1

    def test_train_insufficient_data(self, model):
        prices = pd.DataFrame({"close": [100, 101, 102]})
        model.train(prices, target=None)
        assert model.state.is_trained is False

    def test_train_stores_log_returns(self, model, price_df):
        model.train(price_df, target=None)
        assert model._log_returns is not None
        assert len(model._log_returns) == 199

    def test_train_with_array(self, model):
        prices = np.random.randn(100) + 200
        prices = np.abs(prices)  # ensure positive
        model.train(prices, target=None)
        assert model.state.is_trained is True

    def test_train_filters_non_positive(self, model):
        """Prices with zeros are filtered out."""
        prices = np.array([0, 0, 0] + [100 + i for i in range(60)])
        df = pd.DataFrame({"close": prices})
        model.train(df, target=None)
        assert model.state.is_trained is True


# ── Prediction tests ────────────────────────────────────────────────


class TestGARCHPrediction:
    def test_always_neutral(self, trained_model):
        """GARCH never predicts direction."""
        mock_result = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.variance.values = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
        mock_result.forecast.return_value = mock_forecast

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result

        mock_arch_model = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"arch": MagicMock(arch_model=mock_arch_model)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.NEUTRAL

    def test_magnitude_is_none(self, trained_model):
        """GARCH does not predict magnitude."""
        mock_result = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.variance.values = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
        mock_result.forecast.return_value = mock_forecast

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result
        mock_arch_model = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"arch": MagicMock(arch_model=mock_arch_model)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.magnitude is None

    def test_high_vol_low_confidence(self, trained_model):
        """High volatility forecast should give low confidence."""
        mock_result = MagicMock()
        mock_forecast = MagicMock()
        # Very high variance (scaled returns) -> high annualized vol
        mock_forecast.variance.values = np.array([[100.0, 120.0, 140.0, 150.0, 160.0]])
        mock_result.forecast.return_value = mock_forecast

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result
        mock_arch_model = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"arch": MagicMock(arch_model=mock_arch_model)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        # High vol means lower confidence
        assert result.confidence < 0.5

    def test_low_vol_high_confidence(self, trained_model):
        """Low volatility forecast should give higher confidence."""
        mock_result = MagicMock()
        mock_forecast = MagicMock()
        # Very low variance
        mock_forecast.variance.values = np.array([[0.01, 0.01, 0.01, 0.01, 0.01]])
        mock_result.forecast.return_value = mock_forecast

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result
        mock_arch_model = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"arch": MagicMock(arch_model=mock_arch_model)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.confidence > 0.7

    def test_import_error(self, trained_model):
        with patch.dict("sys.modules", {"arch": None}):
            result = trained_model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_exception(self, trained_model):
        mock_arch_model = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {"arch": MagicMock(arch_model=mock_arch_model)}):
            result = trained_model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_asset_class(self, trained_model):
        result = trained_model.predict(None, symbol="GLD", asset_class=AssetClass.COMMODITY)
        assert result.asset_class == AssetClass.COMMODITY

    def test_predict_horizon_days(self, trained_model):
        result = trained_model.predict(None, horizon_days=20, symbol="AAPL")
        assert result.horizon_days == 20

    def test_predict_model_name(self, trained_model):
        result = trained_model.predict(None, symbol="AAPL")
        assert result.model_name == "garch"
