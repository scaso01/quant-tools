"""LightGBM gradient boosting model for direction + magnitude prediction."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger(__name__)

# Direction thresholds
_UP_THRESHOLD = 0.001
_DOWN_THRESHOLD = -0.001


def _returns_to_labels(returns: np.ndarray) -> np.ndarray:
    """Convert returns to direction labels: 0=DOWN, 1=NEUTRAL, 2=UP."""
    labels = np.ones(len(returns), dtype=int)  # default NEUTRAL
    labels[returns > _UP_THRESHOLD] = 2
    labels[returns < _DOWN_THRESHOLD] = 0
    return labels


class GradientBoostModel(PredictionModel):
    """LightGBM-based direction classifier + magnitude regressor."""

    def __init__(self) -> None:
        super().__init__()
        self._classifier = None
        self._regressor = None
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "gradient_boost"

    def train(self, features_df: object, target: object) -> None:
        """Train classifier and regressor with walk-forward splits."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed; skipping training")
            return

        df = features_df
        y = np.asarray(target).ravel()
        X = np.asarray(df)

        if len(X) < 50:
            logger.warning("Insufficient data for training (%d rows)", len(X))
            return

        from sklearn.model_selection import TimeSeriesSplit

        self._feature_names = list(getattr(df, "columns", []))

        # Direction labels
        y_dir = _returns_to_labels(y)

        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        val_scores: list[float] = []

        params_clf = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
        }
        params_reg = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
        }

        for train_idx, val_idx in tscv.split(X):
            clf = lgb.LGBMClassifier(**params_clf)
            clf.fit(X[train_idx], y_dir[train_idx])
            score = clf.score(X[val_idx], y_dir[val_idx])
            val_scores.append(score)

        # Final model on all data
        self._classifier = lgb.LGBMClassifier(**params_clf)
        self._classifier.fit(X, y_dir)

        self._regressor = lgb.LGBMRegressor(**params_reg)
        self._regressor.fit(X, y)

        self.state.is_trained = True
        self.state.last_trained = datetime.now()
        self.state.training_samples = len(X)
        self.state.validation_score = float(np.mean(val_scores)) if val_scores else 0.0

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Predict direction and magnitude from latest features."""
        if not self.state.is_trained or self._classifier is None:
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )

        try:
            X = np.asarray(features_df)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            last_row = X[[-1]]

            proba = self._classifier.predict_proba(last_row)[0]
            pred_label = int(np.argmax(proba))
            confidence = float(proba[pred_label])

            direction_map = {0: PredictionDirection.DOWN, 1: PredictionDirection.NEUTRAL, 2: PredictionDirection.UP}
            direction = direction_map.get(pred_label, PredictionDirection.NEUTRAL)

            magnitude = float(self._regressor.predict(last_row)[0]) if self._regressor else None

            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=direction,
                magnitude=magnitude,
                confidence=confidence,
                horizon_days=horizon_days,
                interval=interval,
            )
        except Exception:
            logger.exception("GradientBoost prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
