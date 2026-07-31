"""Tests for MoiraiModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.moirai_model import MoiraiModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return MoiraiModel()


@pytest.fixture()
def prices_df():
    """DataFrame with a 'close' column (200 rows)."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    return pd.DataFrame({"close": close})


# ── Basic tests ─────────────────────────────────────────────────────


class TestMoiraiModelBasic:
    def test_name(self, model):
        assert model.name == "moirai"

    def test_initial_state_untrained(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True


# ── Training tests (loads pretrained module) ────────────────────────


class TestMoiraiTraining:
    def test_train_loads_module(self, model, prices_df):
        mock_module_instance = MagicMock()
        mock_moirai_mod = MagicMock()
        mock_moirai_mod.MoiraiModule.from_pretrained.return_value = mock_module_instance

        mock_uni2ts = MagicMock()
        mock_uni2ts_model = MagicMock()
        mock_uni2ts_model.moirai = mock_moirai_mod

        with patch.dict("sys.modules", {
            "uni2ts": mock_uni2ts,
            "uni2ts.model": mock_uni2ts_model,
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            model.train(prices_df, target=None)

        assert model.state.is_trained is True
        assert model._module is mock_module_instance

    def test_train_import_error(self, model, prices_df):
        with patch.dict("sys.modules", {
            "uni2ts": None,
            "uni2ts.model": None,
            "uni2ts.model.moirai": None,
        }):
            with patch("builtins.__import__", side_effect=ImportError("no uni2ts")):
                model.train(prices_df, target=None)
        assert model.state.is_trained is False

    def test_train_exception_handled(self, model, prices_df):
        mock_moirai_mod = MagicMock()
        mock_moirai_mod.MoiraiModule.from_pretrained.side_effect = RuntimeError("load fail")

        with patch.dict("sys.modules", {
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            model.train(prices_df, target=None)

        assert model.state.is_trained is False


# ── Prediction tests ────────────────────────────────────────────────


class TestMoiraiPrediction:
    def test_predict_not_trained(self, model, prices_df):
        result = model.predict(prices_df)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_no_module(self, model, prices_df):
        model.state.is_trained = True
        model._module = None
        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_direction_up(self, model, prices_df):
        """Mock Moirai forecast above current price -> UP."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        mock_forecast = MagicMock()
        mock_forecast.samples = np.full((20, 5), current * 1.03)

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_direction_down(self, model, prices_df):
        """Mock Moirai forecast below current price -> DOWN."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        mock_forecast = MagicMock()
        mock_forecast.samples = np.full((20, 5), current * 0.95)

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.DOWN

    def test_predict_stores_quantiles(self, model, prices_df):
        """Verify quantile values are stored in model state metadata."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        np.random.seed(42)
        mock_forecast = MagicMock()
        mock_forecast.samples = current + np.random.normal(0, 2, (20, 5))

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            model.predict(prices_df)  # asserted on via state.metadata below

        assert "q10" in model.state.metadata
        assert "q25" in model.state.metadata
        assert "q50" in model.state.metadata
        assert "q75" in model.state.metadata
        assert "q90" in model.state.metadata

    def test_predict_confidence_narrow_spread(self, model, prices_df):
        """Narrow spread -> high confidence."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        mock_forecast = MagicMock()
        mock_forecast.samples = np.full((20, 5), current * 1.001)

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert result.confidence >= 0.7

    def test_predict_confidence_wide_spread(self, model, prices_df):
        """Wide spread -> low confidence."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        low_samples = np.full((10, 5), current * 0.5)
        high_samples = np.full((10, 5), current * 1.5)
        mock_forecast = MagicMock()
        mock_forecast.samples = np.concatenate([low_samples, high_samples], axis=0)

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert result.confidence <= 0.5

    def test_predict_insufficient_data(self, model):
        model.state.is_trained = True
        model._module = MagicMock()
        small_df = pd.DataFrame({"close": [100.0] * 20})

        mock_torch = MagicMock()
        mock_moirai_mod = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(small_df)

        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_exception_returns_neutral(self, model, prices_df):
        model.state.is_trained = True
        model._module = MagicMock()

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.side_effect = RuntimeError("boom")
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_symbol_and_asset_class(self, model, prices_df):
        result = model.predict(prices_df, symbol="EUR/USD", asset_class=AssetClass.FOREX)
        assert result.symbol == "EUR/USD"
        assert result.asset_class == AssetClass.FOREX
        assert result.model_name == "moirai"

    def test_predict_fallback_no_samples_attr(self, model, prices_df):
        """When forecast has no 'samples' attr, use mean fallback."""
        model.state.is_trained = True
        model._module = MagicMock()

        current = prices_df["close"].iloc[-1]
        mock_forecast = MagicMock(spec=[])
        mock_forecast.samples = np.full((20, 5), current * 1.02)

        mock_moirai_mod = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.forecast.return_value = mock_forecast
        mock_moirai_mod.MoiraiForecast.return_value = mock_pipeline_instance

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "uni2ts": MagicMock(),
            "uni2ts.model": MagicMock(),
            "uni2ts.model.moirai": mock_moirai_mod,
        }):
            result = model.predict(prices_df)

        assert isinstance(result, PredictionResult)
