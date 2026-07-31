"""Tests for NeuralForecastModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection
from quant_tools.predict.models.neural_forecast import NeuralForecastModel, _MIN_DATA_POINTS


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return NeuralForecastModel()


@pytest.fixture()
def price_df_short():
    """DataFrame with fewer than _MIN_DATA_POINTS rows."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    return pd.DataFrame({"close": prices})


@pytest.fixture()
def price_df_long():
    """DataFrame with enough rows for neural forecast."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    return pd.DataFrame({"close": prices})


@pytest.fixture()
def trained_model(price_df_long):
    m = NeuralForecastModel()
    m.train(price_df_long, target=None)
    return m


# ── Basic tests ─────────────────────────────────────────────────────


class TestNeuralForecastBasic:
    def test_name(self, model):
        assert model.name == "neural_forecast"

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


class TestNeuralForecastTraining:
    def test_train_with_enough_data(self, model, price_df_long):
        model.train(price_df_long, target=None)
        assert model.state.is_trained is True
        assert model.state.training_samples == 300

    def test_train_insufficient_data(self, model, price_df_short):
        model.train(price_df_short, target=None)
        assert model.state.is_trained is False

    def test_train_with_array(self, model):
        prices = np.random.randn(250) + 100
        model.train(prices, target=None)
        assert model.state.is_trained is True

    def test_min_data_points_constant(self):
        assert _MIN_DATA_POINTS == 200


# ── Prediction with mocks ──────────────────────────────────────────


class TestNeuralForecastPrediction:
    def test_predict_up(self, trained_model):
        """Mock neuralforecast to return upward prediction."""
        current = float(trained_model._close_prices[-1])
        up_price = current * 1.05

        mock_nf = MagicMock()
        forecast_df = pd.DataFrame({
            "unique_id": ["asset"] * 5,
            "ds": pd.date_range("2026-01-01", periods=5),
            "LSTM": np.linspace(current, up_price, 5),
        })
        mock_nf.predict.return_value = forecast_df

        mock_nf_class = MagicMock(return_value=mock_nf)
        mock_lstm = MagicMock()

        with patch.dict("sys.modules", {
            "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
            "neuralforecast.models": MagicMock(LSTM=mock_lstm),
        }):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_down(self, trained_model):
        current = float(trained_model._close_prices[-1])
        down_price = current * 0.95

        mock_nf = MagicMock()
        forecast_df = pd.DataFrame({
            "unique_id": ["asset"] * 5,
            "ds": pd.date_range("2026-01-01", periods=5),
            "LSTM": np.linspace(current, down_price, 5),
        })
        mock_nf.predict.return_value = forecast_df
        mock_nf_class = MagicMock(return_value=mock_nf)

        with patch.dict("sys.modules", {
            "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
            "neuralforecast.models": MagicMock(LSTM=MagicMock()),
        }):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.DOWN

    def test_predict_import_error(self, trained_model):
        with patch.dict("sys.modules", {
            "neuralforecast": None,
            "neuralforecast.models": None,
        }):
            result = trained_model.predict(None, symbol="AAPL")

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_insufficient_data_after_train(self):
        """Model trained but data below minimum."""
        model = NeuralForecastModel()
        model.state.is_trained = True
        model._close_prices = np.random.randn(50) + 100
        result = model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_exception(self, trained_model):
        mock_nf_class = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {
            "neuralforecast": MagicMock(NeuralForecast=mock_nf_class),
            "neuralforecast.models": MagicMock(LSTM=MagicMock()),
        }):
            result = trained_model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_horizon_days(self, trained_model):
        result = trained_model.predict(None, horizon_days=20, symbol="AAPL")
        assert result.horizon_days == 20

    def test_predict_model_name(self, trained_model):
        result = trained_model.predict(None, symbol="AAPL")
        assert result.model_name == "neural_forecast"
