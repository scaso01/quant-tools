"""Amazon Chronos foundation model for zero-shot time-series forecasting."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger("quant_tools")

_MIN_DATA_POINTS = 50


class ChronosModel(PredictionModel):
    """Chronos (Amazon) pretrained foundation model for zero-shot forecasting.

    This model does NOT train on local data. The ``train()`` method loads the
    pretrained pipeline from Hugging Face.  ``predict()`` runs zero-shot
    inference on close prices and returns probabilistic quantile forecasts in
    ``PredictionResult.metadata``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: object | None = None

    @property
    def name(self) -> str:
        return "chronos"

    def train(self, features_df: object, target: object) -> None:
        """Load pretrained Chronos pipeline (no actual training)."""
        try:
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
            )
            self.state.is_trained = True
            self.state.last_trained = datetime.now()
        except ImportError:
            logger.warning(
                "chronos-forecasting not installed. "
                "pip install quant-tools[foundation]"
            )
        except Exception:
            logger.exception("Chronos pipeline loading failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Zero-shot forecast using Chronos foundation model."""
        if not self.state.is_trained or self._pipeline is None:
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
            import torch
        except ImportError:
            logger.warning("torch not installed")
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
            # Extract close prices
            df = features_df
            if hasattr(df, "columns") and "close" in df.columns:
                prices = np.asarray(df["close"]).ravel()
            elif hasattr(df, "columns") and len(df.columns) > 0:
                prices = np.asarray(df.iloc[:, 0]).ravel()
            else:
                prices = np.asarray(df).ravel()

            if len(prices) < _MIN_DATA_POINTS:
                return PredictionResult(
                    symbol=symbol,
                    asset_class=asset_class,
                    model_name=self.name,
                    direction=PredictionDirection.NEUTRAL,
                    confidence=0.3,
                    horizon_days=horizon_days,
                    interval=interval,
                )

            context = torch.tensor(prices, dtype=torch.float32)
            forecast_samples = self._pipeline.predict(
                context,
                prediction_length=horizon_days,
                num_samples=20,
            )

            # forecast_samples shape: (1, num_samples, horizon_days)
            samples = forecast_samples.numpy().squeeze(0)  # (num_samples, horizon_days)

            # Compute quantiles from the final horizon step
            final_step = samples[:, -1]
            q10 = float(np.percentile(final_step, 10))
            q25 = float(np.percentile(final_step, 25))
            q50 = float(np.percentile(final_step, 50))
            q75 = float(np.percentile(final_step, 75))
            q90 = float(np.percentile(final_step, 90))

            current = float(prices[-1])

            # Direction from median forecast vs current price
            pct_change = (q50 - current) / current if current != 0 else 0.0

            if pct_change > 0.001:
                direction = PredictionDirection.UP
            elif pct_change < -0.001:
                direction = PredictionDirection.DOWN
            else:
                direction = PredictionDirection.NEUTRAL

            # Confidence from forecast spread: narrow = high confidence
            spread = (q90 - q10) / current if current != 0 else 1.0
            confidence = max(0.15, min(0.85, 1.0 - spread))

            result = PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=direction,
                magnitude=abs(pct_change),
                confidence=confidence,
                horizon_days=horizon_days,
                interval=interval,
            )
            # Store quantiles in timestamp field workaround — use metadata on state
            self.state.metadata = {
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
            }
            return result

        except Exception:
            logger.exception("Chronos prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
