"""GARCH(1,1) volatility forecasting model."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger(__name__)


class GARCHModel(PredictionModel):
    """GARCH(1,1) model for volatility forecasting.

    This model does NOT predict direction. It always returns NEUTRAL
    and provides volatility-based confidence for the ensemble's risk adjustment.
    """

    def __init__(self) -> None:
        super().__init__()
        self._log_returns: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "garch"

    def train(self, features_df: object, target: object) -> None:
        """Store log returns for GARCH fitting."""
        try:
            df = features_df
            if hasattr(df, "columns") and "close" in df.columns:
                prices = np.asarray(df["close"]).ravel()
            elif hasattr(df, "columns") and len(df.columns) > 0:
                prices = np.asarray(df.iloc[:, 0]).ravel()
            else:
                prices = np.asarray(df).ravel()

            if len(prices) < 50:
                logger.warning("Insufficient data for GARCH (%d rows)", len(prices))
                return

            # Compute log returns
            prices_clean = prices[prices > 0]
            if len(prices_clean) < 50:
                return
            self._log_returns = np.diff(np.log(prices_clean))

            self.state.is_trained = True
            self.state.last_trained = datetime.now()
            self.state.training_samples = len(self._log_returns)
        except Exception:
            logger.exception("GARCH training failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Forecast volatility using GARCH(1,1). Always returns NEUTRAL direction."""
        if not self.state.is_trained or self._log_returns is None:
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
            from arch import arch_model
        except ImportError:
            self.state.is_trained = False
            logger.warning("arch package not installed")
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
            # Scale returns by 100 for numerical stability
            returns_scaled = self._log_returns * 100

            model = arch_model(returns_scaled, vol="GARCH", p=1, q=1, mean="Zero")
            result = model.fit(disp="off")

            # Forecast variance
            forecast = result.forecast(horizon=horizon_days)
            variance = forecast.variance.values[-1]
            mean_variance = float(np.mean(variance))

            # Convert back from scaled
            annualized_vol = float(np.sqrt(mean_variance / 10000 * 252))

            # Confidence: inverse of volatility (high vol = low confidence)
            # Typical annual vol range: 0.10 (low) to 0.60 (high)
            confidence = max(0.15, min(0.85, 1.0 - annualized_vol))

            result_obj = PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                magnitude=None,
                confidence=confidence,
                horizon_days=horizon_days,
                interval=interval,
            )
            # Store volatility forecast for ensemble risk adjustment
            result_obj.timestamp = datetime.now()
            return result_obj

        except Exception:
            logger.exception("GARCH prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
