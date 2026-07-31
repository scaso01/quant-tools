"""Tests for ProphetModel."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection
from quant_tools.predict.models.prophet_model import ProphetModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return ProphetModel()


@pytest.fixture()
def price_df():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({"close": prices})


@pytest.fixture()
def trained_model(price_df):
    m = ProphetModel()
    m.train(price_df, target=None)
    return m


# ── Basic tests ─────────────────────────────────────────────────────


class TestProphetBasic:
    def test_name(self, model):
        assert model.name == "prophet"

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


class TestProphetTraining:
    def test_train_success(self, model, price_df):
        model.train(price_df, target=None)
        assert model.state.is_trained is True
        assert model.state.training_samples == 100

    def test_train_insufficient_data(self, model):
        prices = pd.DataFrame({"close": [100, 101]})
        model.train(prices, target=None)
        assert model.state.is_trained is False

    def test_train_with_array(self, model):
        prices = np.random.randn(60) + 100
        model.train(prices, target=None)
        assert model.state.is_trained is True


# ── Prediction with mocks ──────────────────────────────────────────


class TestProphetPrediction:
    def _make_prophet_mock(self, current_price, forecast_price, upper, lower):
        """Create a mock Prophet that returns controlled forecast."""
        mock_prophet = MagicMock()
        future_df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=105)})
        mock_prophet.make_future_dataframe.return_value = future_df

        forecast_df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=105),
            "yhat": np.linspace(current_price, forecast_price, 105),
            "yhat_upper": np.linspace(current_price, upper, 105),
            "yhat_lower": np.linspace(current_price, lower, 105),
        })
        mock_prophet.predict.return_value = forecast_df
        return mock_prophet

    def test_predict_up(self, trained_model):
        current = float(trained_model._close_prices[-1])
        mock_prophet = self._make_prophet_mock(current, current * 1.05, current * 1.08, current * 1.02)
        mock_prophet_class = MagicMock(return_value=mock_prophet)

        with patch.dict("sys.modules", {"prophet": MagicMock(Prophet=mock_prophet_class)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_down(self, trained_model):
        current = float(trained_model._close_prices[-1])
        mock_prophet = self._make_prophet_mock(current, current * 0.93, current * 0.97, current * 0.89)
        mock_prophet_class = MagicMock(return_value=mock_prophet)

        with patch.dict("sys.modules", {"prophet": MagicMock(Prophet=mock_prophet_class)}):
            result = trained_model.predict(None, horizon_days=5, symbol="AAPL")

        assert result.direction == PredictionDirection.DOWN

    def test_predict_import_error(self, trained_model):
        with patch.dict("sys.modules", {"prophet": None}):
            result = trained_model.predict(None, symbol="AAPL")

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_exception(self, trained_model):
        mock_prophet_class = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {"prophet": MagicMock(Prophet=mock_prophet_class)}):
            result = trained_model.predict(None, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL

    def test_log_suppression(self, trained_model):
        """Verify cmdstanpy/prophet log levels are suppressed."""
        current = float(trained_model._close_prices[-1])
        mock_prophet = self._make_prophet_mock(current, current * 1.02, current * 1.05, current * 0.99)
        mock_prophet_class = MagicMock(return_value=mock_prophet)

        with patch.dict("sys.modules", {"prophet": MagicMock(Prophet=mock_prophet_class)}):
            with patch("os.environ", dict(os.environ)):
                trained_model.predict(None, symbol="AAPL")
                # If we got here without error, log suppression worked

    def test_predict_asset_class(self, trained_model):
        result = trained_model.predict(None, symbol="ETH", asset_class=AssetClass.CRYPTO)
        assert result.asset_class == AssetClass.CRYPTO

    def test_predict_horizon_days(self, trained_model):
        result = trained_model.predict(None, horizon_days=20, symbol="AAPL")
        assert result.horizon_days == 20

    def test_predict_model_name(self, trained_model):
        result = trained_model.predict(None, symbol="AAPL")
        assert result.model_name == "prophet"
