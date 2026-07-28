"""Tests for ChronosModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.chronos_model import ChronosModel


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return ChronosModel()


@pytest.fixture()
def prices_df():
    """DataFrame with a 'close' column (200 rows)."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    return pd.DataFrame({"close": close})


# ── Basic tests ─────────────────────────────────────────────────────


class TestChronosModelBasic:
    def test_name(self, model):
        assert model.name == "chronos"

    def test_initial_state_untrained(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True


# ── Training tests (loads pretrained pipeline) ──────────────────────


class TestChronosTraining:
    def test_train_loads_pipeline(self, model, prices_df):
        mock_pipeline = MagicMock()
        mock_chronos_mod = MagicMock()
        mock_chronos_mod.ChronosPipeline.from_pretrained.return_value = mock_pipeline

        with patch.dict("sys.modules", {"chronos": mock_chronos_mod}):
            model.train(prices_df, target=None)

        assert model.state.is_trained is True
        assert model._pipeline is mock_pipeline
        mock_chronos_mod.ChronosPipeline.from_pretrained.assert_called_once_with(
            "amazon/chronos-t5-small",
            device_map="cpu",
        )

    def test_train_import_error(self, model, prices_df):
        with patch.dict("sys.modules", {"chronos": None}):
            with patch("builtins.__import__", side_effect=ImportError("no chronos")):
                model.train(prices_df, target=None)
        assert model.state.is_trained is False

    def test_train_exception_handled(self, model, prices_df):
        mock_chronos_mod = MagicMock()
        mock_chronos_mod.ChronosPipeline.from_pretrained.side_effect = RuntimeError("load fail")

        with patch.dict("sys.modules", {"chronos": mock_chronos_mod}):
            model.train(prices_df, target=None)

        assert model.state.is_trained is False


# ── Prediction tests ────────────────────────────────────────────────


class TestChronosPrediction:
    def test_predict_not_trained(self, model, prices_df):
        result = model.predict(prices_df)
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_no_pipeline(self, model, prices_df):
        model.state.is_trained = True
        model._pipeline = None
        result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_quantiles_direction_up(self, model, prices_df):
        """Mock pipeline to return samples above current price -> UP."""
        model.state.is_trained = True

        current = prices_df["close"].iloc[-1]
        # Samples above current price
        mock_samples = np.array([[[current * 1.02] * 5] * 20])  # (1, 20, 5)

        mock_pipeline = MagicMock()
        mock_tensor_result = MagicMock()
        mock_tensor_result.numpy.return_value = mock_samples
        mock_pipeline.predict.return_value = mock_tensor_result
        model._pipeline = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.UP
        assert result.magnitude > 0

    def test_predict_quantiles_direction_down(self, model, prices_df):
        """Mock pipeline to return samples below current price -> DOWN."""
        model.state.is_trained = True

        current = prices_df["close"].iloc[-1]
        mock_samples = np.array([[[current * 0.95] * 5] * 20])

        mock_pipeline = MagicMock()
        mock_tensor_result = MagicMock()
        mock_tensor_result.numpy.return_value = mock_samples
        mock_pipeline.predict.return_value = mock_tensor_result
        model._pipeline = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.DOWN

    def test_predict_stores_quantiles(self, model, prices_df):
        """Verify quantile values are stored in model state metadata."""
        model.state.is_trained = True

        current = prices_df["close"].iloc[-1]
        # Spread of samples for varied quantiles
        np.random.seed(42)
        samples = current + np.random.normal(0, 2, (1, 20, 5))

        mock_pipeline = MagicMock()
        mock_tensor_result = MagicMock()
        mock_tensor_result.numpy.return_value = samples
        mock_pipeline.predict.return_value = mock_tensor_result
        model._pipeline = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert "q10" in model.state.metadata
        assert "q25" in model.state.metadata
        assert "q50" in model.state.metadata
        assert "q75" in model.state.metadata
        assert "q90" in model.state.metadata

    def test_predict_confidence_narrow_spread(self, model, prices_df):
        """Narrow spread in samples -> high confidence."""
        model.state.is_trained = True

        current = prices_df["close"].iloc[-1]
        # Very narrow spread: all samples nearly identical
        samples = np.array([[[current * 1.001] * 5] * 20])

        mock_pipeline = MagicMock()
        mock_tensor_result = MagicMock()
        mock_tensor_result.numpy.return_value = samples
        mock_pipeline.predict.return_value = mock_tensor_result
        model._pipeline = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert result.confidence >= 0.7

    def test_predict_confidence_wide_spread(self, model, prices_df):
        """Wide spread in samples -> low confidence."""
        model.state.is_trained = True

        current = prices_df["close"].iloc[-1]
        # Very wide spread
        low_samples = np.full((10, 5), current * 0.5)
        high_samples = np.full((10, 5), current * 1.5)
        samples = np.concatenate([low_samples, high_samples], axis=0)
        samples = samples[np.newaxis, :, :]  # (1, 20, 5)

        mock_pipeline = MagicMock()
        mock_tensor_result = MagicMock()
        mock_tensor_result.numpy.return_value = samples
        mock_pipeline.predict.return_value = mock_tensor_result
        model._pipeline = mock_pipeline

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert result.confidence <= 0.5

    def test_predict_insufficient_data(self, model):
        model.state.is_trained = True
        model._pipeline = MagicMock()
        small_df = pd.DataFrame({"close": [100.0] * 20})

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(small_df)

        assert result.direction == PredictionDirection.NEUTRAL

    @patch.dict("sys.modules", {"torch": None})
    def test_predict_torch_import_error(self, model, prices_df):
        model.state.is_trained = True
        model._pipeline = MagicMock()
        with patch("builtins.__import__", side_effect=ImportError("no torch")):
            result = model.predict(prices_df)
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_exception_returns_neutral(self, model, prices_df):
        model.state.is_trained = True
        model._pipeline = MagicMock()
        model._pipeline.predict.side_effect = RuntimeError("boom")

        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = model.predict(prices_df)

        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_symbol_and_asset_class(self, model, prices_df):
        result = model.predict(prices_df, symbol="BTC", asset_class=AssetClass.CRYPTO)
        assert result.symbol == "BTC"
        assert result.asset_class == AssetClass.CRYPTO
        assert result.model_name == "chronos"
