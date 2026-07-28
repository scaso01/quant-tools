"""Tests for TFTModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.tft_model import TFTModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return TFTModel()


@pytest.fixture()
def prices_df():
    """DataFrame with a 'close' column (200 rows)."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    return pd.DataFrame({"close": close})


# ── Basic tests ─────────────────────────────────────────────────────


class TestTFTModelBasic:
    def test_name(self, model):
        assert model.name == "tft"

    def test_initial_state_untrained(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True


# ── Training tests ──────────────────────────────────────────────────


class TestTFTTraining:
    def test_train_stores_prices(self, model, prices_df):
        model.train(prices_df, target=None)
        assert model.state.is_trained is True
        assert model.state.training_samples == 200
        assert model._close_prices is not None

    def test_train_insufficient_data(self, model):
        small_df = pd.DataFrame({"close": [100.0] * 20})
        model.train(small_df, target=None)
        assert model.state.is_trained is False

    def test_train_from_array(self, model):
        arr = np.random.randn(100) + 100
        model.train(arr, target=None)
        assert model.state.is_trained is True

    def test_train_from_df_no_close_column(self, model):
        df = pd.DataFrame({"price": np.random.randn(100) + 100})
        model.train(df, target=None)
        assert model.state.is_trained is True


# ── Prediction tests ────────────────────────────────────────────────


class TestTFTPrediction:
    def test_predict_not_trained(self, model, prices_df):
        result = model.predict(prices_df)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_insufficient_data(self, model):
        model.state.is_trained = True
        model._close_prices = np.array([100.0] * 20)
        result = model.predict(pd.DataFrame({"close": [100.0] * 20}))
        assert result.direction == PredictionDirection.NEUTRAL

    @patch.dict("sys.modules", {"neuralforecast": None, "neuralforecast.models": None})
    def test_predict_import_error(self, model, prices_df):
        model.train(prices_df, target=None)
        with patch("builtins.__import__", side_effect=ImportError("no neuralforecast")):
            result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_direction_up(self, model, prices_df):
        """Mock NeuralForecast to return forecast above current price."""
        model.train(prices_df, target=None)
        current = model._close_prices[-1]

        mock_forecast_df = pd.DataFrame({"TFT": [current * 1.05] * 5})
        mock_nf_instance = MagicMock()
        mock_nf_instance.predict.return_value = mock_forecast_df

        mock_tft_class = MagicMock()
        mock_nf_class = MagicMock(return_value=mock_nf_instance)

        mock_nf_mod = MagicMock()
        mock_nf_mod.NeuralForecast = mock_nf_class
        mock_nf_models_mod = MagicMock()
        mock_nf_models_mod.TFT = mock_tft_class

        with patch.dict("sys.modules", {
            "neuralforecast": mock_nf_mod,
            "neuralforecast.models": mock_nf_models_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_direction_down(self, model, prices_df):
        """Mock NeuralForecast to return forecast below current price."""
        model.train(prices_df, target=None)
        current = model._close_prices[-1]

        mock_forecast_df = pd.DataFrame({"TFT": [current * 0.95] * 5})
        mock_nf_instance = MagicMock()
        mock_nf_instance.predict.return_value = mock_forecast_df

        mock_tft_class = MagicMock()
        mock_nf_class = MagicMock(return_value=mock_nf_instance)

        mock_nf_mod = MagicMock()
        mock_nf_mod.NeuralForecast = mock_nf_class
        mock_nf_models_mod = MagicMock()
        mock_nf_models_mod.TFT = mock_tft_class

        with patch.dict("sys.modules", {
            "neuralforecast": mock_nf_mod,
            "neuralforecast.models": mock_nf_models_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.DOWN

    def test_predict_symbol_and_asset_class(self, model, prices_df):
        result = model.predict(prices_df, symbol="AAPL", asset_class=AssetClass.EQUITY)
        assert result.symbol == "AAPL"
        assert result.asset_class == AssetClass.EQUITY
        assert result.model_name == "tft"

    def test_predict_horizon(self, model, prices_df):
        result = model.predict(prices_df, horizon_days=20)
        assert result.horizon_days == 20

    def test_predict_exception_returns_neutral(self, model, prices_df):
        """If the NF predict call raises, return NEUTRAL gracefully."""
        model.train(prices_df, target=None)

        mock_nf_instance = MagicMock()
        mock_nf_instance.predict.side_effect = RuntimeError("boom")

        mock_nf_class = MagicMock(return_value=mock_nf_instance)
        mock_tft_class = MagicMock()

        mock_nf_mod = MagicMock()
        mock_nf_mod.NeuralForecast = mock_nf_class
        mock_nf_models_mod = MagicMock()
        mock_nf_models_mod.TFT = mock_tft_class

        with patch.dict("sys.modules", {
            "neuralforecast": mock_nf_mod,
            "neuralforecast.models": mock_nf_models_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3
