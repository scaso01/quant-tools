"""Tests for StatsForecastModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.stats_forecast import StatsForecastModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return StatsForecastModel()


@pytest.fixture()
def price_df():
    """DataFrame with close prices."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"close": prices})


@pytest.fixture()
def trained_model(price_df):
    m = StatsForecastModel()
    m.train(price_df, target=None)
    return m


# ── Basic tests ─────────────────────────────────────────────────────


class TestStatsForecastBasic:
    def test_name(self, model):
        assert model.name == "stats_forecast"

    def test_initial_state(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True

    def test_predict_without_training(self, model, price_df):
        result = model.predict(price_df, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3


# ── Training tests ──────────────────────────────────────────────────


class TestStatsForecastTraining:
    def test_train_with_close_column(self, model, price_df):
        model.train(price_df, target=None)
        assert model.state.is_trained is True
        assert model.state.training_samples == 100

    def test_train_with_array(self, model):
        prices = np.random.randn(60) + 100
        model.train(prices, target=None)
        assert model.state.is_trained is True

    def test_train_insufficient_data(self, model):
        prices = pd.DataFrame({"close": [100, 101, 102]})
        model.train(prices, target=None)
        assert model.state.is_trained is False

    def test_train_no_close_column(self, model):
        df = pd.DataFrame({"price": np.random.randn(50) + 100})
        model.train(df, target=None)
        assert model.state.is_trained is True


# ── Prediction with mocks ──────────────────────────────────────────


class TestStatsForecastPrediction:
    def test_predict_up(self, trained_model):
        """Mock statsforecast to return an upward forecast."""
        mock_sf = MagicMock()
        forecast_df = pd.DataFrame({
            "unique_id": ["asset"] * 5,
            "ds": pd.date_range("2026-01-01", periods=5),
            "AutoARIMA": [105.0, 106.0, 107.0, 108.0, 110.0],
            "AutoARIMA-lo-90": [103.0, 104.0, 105.0, 106.0, 107.0],
            "AutoARIMA-hi-90": [107.0, 108.0, 109.0, 110.0, 113.0],
        })
        mock_sf.predict.return_value = forecast_df

        mock_sf_class = MagicMock(return_value=mock_sf)
        mock_arima = MagicMock()

        with patch.dict("sys.modules", {
            "statsforecast": MagicMock(StatsForecast=mock_sf_class),
            "statsforecast.models": MagicMock(AutoARIMA=mock_arima),
        }):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_down(self, trained_model):
        """Mock statsforecast to return a downward forecast."""
        current_price = float(trained_model._close_prices[-1])
        low_price = current_price * 0.95

        mock_sf = MagicMock()
        forecast_df = pd.DataFrame({
            "unique_id": ["asset"] * 5,
            "ds": pd.date_range("2026-01-01", periods=5),
            "AutoARIMA": [current_price - 1, current_price - 2, current_price - 3, current_price - 4, low_price],
            "AutoARIMA-lo-90": [low_price - 5] * 5,
            "AutoARIMA-hi-90": [current_price] * 5,
        })
        mock_sf.predict.return_value = forecast_df

        mock_sf_class = MagicMock(return_value=mock_sf)

        with patch.dict("sys.modules", {
            "statsforecast": MagicMock(StatsForecast=mock_sf_class),
            "statsforecast.models": MagicMock(AutoARIMA=MagicMock()),
        }):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.DOWN

    def test_predict_import_error(self, trained_model):
        """ImportError returns NEUTRAL."""
        with patch.dict("sys.modules", {
            "statsforecast": None,
            "statsforecast.models": None,
        }):
            result = trained_model.predict(None, symbol="AAPL")

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_exception(self, trained_model):
        """Runtime exception returns NEUTRAL."""
        mock_sf_class = MagicMock(side_effect=RuntimeError("boom"))

        with patch.dict("sys.modules", {
            "statsforecast": MagicMock(StatsForecast=mock_sf_class),
            "statsforecast.models": MagicMock(AutoARIMA=MagicMock()),
        }):
            result = trained_model.predict(None, symbol="AAPL")

        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_asset_class(self, trained_model):
        result = trained_model.predict(None, symbol="BTC", asset_class=AssetClass.CRYPTO)
        assert result.asset_class == AssetClass.CRYPTO

    def test_predict_horizon_days(self, trained_model):
        result = trained_model.predict(None, horizon_days=20, symbol="AAPL")
        assert result.horizon_days == 20

    def test_predict_model_name(self, trained_model):
        result = trained_model.predict(None, symbol="AAPL")
        assert result.model_name == "stats_forecast"
