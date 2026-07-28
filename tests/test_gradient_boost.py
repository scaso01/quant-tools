"""Tests for GradientBoostModel."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.models.gradient_boost import GradientBoostModel, _returns_to_labels


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def model():
    return GradientBoostModel()


@pytest.fixture()
def synthetic_features():
    """Synthetic feature matrix (100 rows, 5 features)."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture()
def synthetic_target():
    """Synthetic return target."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02


# ── Helper tests ────────────────────────────────────────────────────


class TestReturnsToLabels:
    def test_up_label(self):
        labels = _returns_to_labels(np.array([0.01]))
        assert labels[0] == 2

    def test_down_label(self):
        labels = _returns_to_labels(np.array([-0.01]))
        assert labels[0] == 0

    def test_neutral_label(self):
        labels = _returns_to_labels(np.array([0.0005]))
        assert labels[0] == 1

    def test_mixed(self):
        ret = np.array([0.05, -0.03, 0.0001, 0.002, -0.005])
        labels = _returns_to_labels(ret)
        assert list(labels) == [2, 0, 1, 2, 0]


# ── Model basic tests ──────────────────────────────────────────────


class TestGradientBoostModelBasic:
    def test_name(self, model):
        assert model.name == "gradient_boost"

    def test_initial_state_untrained(self, model):
        assert model.state.is_trained is False

    def test_supports_all_asset_classes(self, model):
        for ac in AssetClass:
            assert model.supports_asset_class(ac) is True

    def test_predict_without_training(self, model, synthetic_features):
        result = model.predict(synthetic_features, horizon_days=5, symbol="AAPL")
        assert isinstance(result, PredictionResult)
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3
        assert result.model_name == "gradient_boost"

    def test_predict_returns_prediction_result(self, model, synthetic_features):
        result = model.predict(synthetic_features, symbol="SPY")
        assert isinstance(result, PredictionResult)
        assert result.symbol == "SPY"


# ── Training with mocked LightGBM ──────────────────────────────────


class TestGradientBoostTraining:
    @patch("quant_tools.predict.models.gradient_boost.lgb", create=True)
    def test_train_updates_state(self, mock_lgb_mod, model, synthetic_features, synthetic_target):
        """Train with mocked lightgbm classes."""
        mock_clf = MagicMock()
        mock_clf.score.return_value = 0.65
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([0.015])

        mock_lgbm_clf_class = MagicMock(return_value=mock_clf)
        mock_lgbm_reg_class = MagicMock(return_value=mock_reg)

        with patch.dict("sys.modules", {"lightgbm": MagicMock(
            LGBMClassifier=mock_lgbm_clf_class,
            LGBMRegressor=mock_lgbm_reg_class,
        )}):
            model.train(synthetic_features, synthetic_target)

        assert model.state.is_trained is True
        assert model.state.training_samples == 100

    @patch.dict("sys.modules", {"lightgbm": None})
    def test_train_import_error(self, model, synthetic_features, synthetic_target):
        """ImportError during training is handled gracefully."""
        # lightgbm set to None in sys.modules triggers ImportError
        with patch("builtins.__import__", side_effect=ImportError("no lightgbm")):
            model.train(synthetic_features, synthetic_target)
        assert model.state.is_trained is False

    def test_train_insufficient_data(self, model):
        """Too few rows skips training."""
        small_X = np.random.randn(10, 3)
        small_y = np.random.randn(10)
        with patch.dict("sys.modules", {"lightgbm": MagicMock()}):
            model.train(small_X, small_y)
        assert model.state.is_trained is False


# ── Prediction with mocked LightGBM ────────────────────────────────


class TestGradientBoostPrediction:
    def _make_trained_model(self):
        model = GradientBoostModel()
        model.state.is_trained = True
        model.state.training_samples = 100

        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.15, 0.75]])
        model._classifier = mock_clf

        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([0.02])
        model._regressor = mock_reg

        return model

    def test_predict_up(self):
        model = self._make_trained_model()
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="AAPL")
        assert result.direction == PredictionDirection.UP
        assert result.confidence == pytest.approx(0.75)
        assert result.magnitude == pytest.approx(0.02)

    def test_predict_down(self):
        model = self._make_trained_model()
        model._classifier.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="TSLA")
        assert result.direction == PredictionDirection.DOWN
        assert result.confidence == pytest.approx(0.8)

    def test_predict_neutral(self):
        model = self._make_trained_model()
        model._classifier.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="SPY")
        assert result.direction == PredictionDirection.NEUTRAL

    def test_predict_1d_input(self):
        model = self._make_trained_model()
        features = np.random.randn(5)  # 1D array
        result = model.predict(features, symbol="MSFT")
        assert isinstance(result, PredictionResult)

    def test_predict_asset_class(self):
        model = self._make_trained_model()
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="BTC-USD", asset_class=AssetClass.CRYPTO)
        assert result.asset_class == AssetClass.CRYPTO

    def test_predict_horizon_days(self):
        model = self._make_trained_model()
        features = np.random.randn(1, 5)
        result = model.predict(features, horizon_days=20, symbol="AAPL")
        assert result.horizon_days == 20

    def test_predict_exception_returns_neutral(self):
        model = self._make_trained_model()
        model._classifier.predict_proba.side_effect = RuntimeError("boom")
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="AAPL")
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.confidence == 0.3

    def test_predict_no_regressor(self):
        model = self._make_trained_model()
        model._regressor = None
        features = np.random.randn(1, 5)
        result = model.predict(features, symbol="AAPL")
        assert result.magnitude is None

    def test_predict_default_symbol(self):
        model = self._make_trained_model()
        features = np.random.randn(1, 5)
        result = model.predict(features)
        assert result.symbol == "UNKNOWN"
