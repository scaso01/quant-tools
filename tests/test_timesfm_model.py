"""Tests for TimesFMModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.timesfm_model import TimesFMModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return TimesFMModel()


@pytest.fixture()
def prices_df():
    """DataFrame with a 'close' column (200 rows)."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    return pd.DataFrame({"close": close})


# ── Basic tests ─────────────────────────────────────────────────────


class TestTimesFMModelBasic:
    def test_name(self, model):
        assert model.name == "timesfm"

    def test_initial_state_untrained(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True


# ── Training tests (loads pretrained checkpoint) ────────────────────


class TestTimesFMTraining:
    def test_train_loads_model(self, model, prices_df):
        mock_timesfm_mod = MagicMock()
        mock_model_instance = MagicMock()
        mock_timesfm_mod.TimesFm.return_value = mock_model_instance

        with patch.dict("sys.modules", {"timesfm": mock_timesfm_mod}):
            model.train(prices_df, target=None)

        assert model.state.is_trained is True
        assert model._model is mock_model_instance

    def test_train_import_error(self, model, prices_df):
        with patch.dict("sys.modules", {"timesfm": None}):
            with patch("builtins.__import__", side_effect=ImportError("no timesfm")):
                model.train(prices_df, target=None)
        assert model.state.is_trained is False

    def test_train_exception_handled(self, model, prices_df):
        mock_timesfm_mod = MagicMock()
        mock_timesfm_mod.TimesFm.side_effect = RuntimeError("load fail")

        with patch.dict("sys.modules", {"timesfm": mock_timesfm_mod}):
            model.train(prices_df, target=None)

        assert model.state.is_trained is False


# ── Prediction tests ────────────────────────────────────────────────


class TestTimesFMPrediction:
    def test_predict_not_trained(self, model, prices_df):
        result = model.predict(prices_df)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_no_model(self, model, prices_df):
        model.state.is_trained = True
        model._model = None
        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_direction_up(self, model, prices_df):
        """Mock forecast above current price -> UP."""
        model.state.is_trained = True
        current = prices_df["close"].iloc[-1]

        mock_model = MagicMock()
        # point_forecast: (1, horizon_len)
        mock_model.forecast.return_value = (
            np.array([[current * 1.03] * 128]),  # point forecast
            np.array([[[current * 1.01, current * 1.02, current * 1.03,
                        current * 1.04, current * 1.05]] * 128]),  # quantile forecast
        )
        model._model = mock_model

        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_direction_down(self, model, prices_df):
        """Mock forecast below current price -> DOWN."""
        model.state.is_trained = True
        current = prices_df["close"].iloc[-1]

        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.array([[current * 0.95] * 128]),
            None,  # No quantile forecast
        )
        model._model = mock_model

        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.DOWN

    def test_predict_stores_quantiles(self, model, prices_df):
        """Verify quantile values are stored in model state metadata."""
        model.state.is_trained = True
        current = prices_df["close"].iloc[-1]

        q_values = [current * 0.98, current * 0.99, current * 1.0,
                     current * 1.01, current * 1.02]
        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.array([[current * 1.0] * 128]),
            np.array([[q_values] * 128]),
        )
        model._model = mock_model

        model.predict(prices_df)  # asserted on via state.metadata below

        assert "q10" in model.state.metadata
        assert "q50" in model.state.metadata
        assert "q90" in model.state.metadata

    def test_predict_no_quantile_forecast(self, model, prices_df):
        """When quantile forecast is None, fall back to point forecast."""
        model.state.is_trained = True
        current = prices_df["close"].iloc[-1]

        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.array([[current * 1.02] * 128]),
            None,
        )
        model._model = mock_model

        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.UP

    def test_predict_insufficient_data(self, model):
        model.state.is_trained = True
        model._model = MagicMock()
        small_df = pd.DataFrame({"close": [100.0] * 20})
        result = model.predict(small_df)
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_exception_returns_neutral(self, model, prices_df):
        model.state.is_trained = True
        model._model = MagicMock()
        model._model.forecast.side_effect = RuntimeError("boom")
        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_symbol_and_asset_class(self, model, prices_df):
        result = model.predict(prices_df, symbol="GLD", asset_class=AssetClass.COMMODITY)
        assert result.symbol == "GLD"
        assert result.asset_class == AssetClass.COMMODITY
        assert result.model_name == "timesfm"

    def test_predict_horizon(self, model, prices_df):
        result = model.predict(prices_df, horizon_days=20)
        assert result.horizon_days == 20
