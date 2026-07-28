"""Salesforce Moirai foundation model for zero-shot time-series forecasting."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger("quant_tools")

_MIN_DATA_POINTS = 50


class MoiraiModel(PredictionModel):
    """Moirai (Salesforce) pretrained foundation model for zero-shot forecasting.

    This model does NOT train on local data. The ``train()`` method loads the
    pretrained module from Hugging Face.  ``predict()`` runs zero-shot
    inference on close prices and returns probabilistic quantile forecasts in
    ``PredictionResult.metadata``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._module: object | None = None

    @property
    def name(self) -> str:
        return "moirai"

    def train(self, features_df: object, target: object) -> None:
        """Load Moirai pretrained module (no actual training)."""
        try:
            from uni2ts.model.moirai import MoiraiModule

            self._module = MoiraiModule.from_pretrained(
                "Salesforce/moirai-1.1-R-small",
            )
            self.state.is_trained = True
            self.state.last_trained = datetime.now()
        except ImportError:
            logger.warning(
                "uni2ts not installed. pip install quant-tools[foundation]"
            )
        except Exception:
            logger.exception("Moirai module loading failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Zero-shot forecast using Moirai foundation model."""
        if not self.state.is_trained or self._module is None:
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
            from uni2ts.model.moirai import MoiraiForecast
        except ImportError:
            logger.warning("uni2ts or torch not installed")
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
            import pandas as pd

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

            # Frequency mapping
            freq_map = {"1d": "D", "1h": "h", "4h": "4h", "15m": "15min"}
            freq = freq_map.get(interval, "D")

            # Build time series DataFrame
            n = len(prices)
            ds = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="B")
            ts_df = pd.DataFrame({"target": prices.astype(float)}, index=ds)

            # Create forecast pipeline
            pipeline = MoiraiForecast(
                module=self._module,
                prediction_length=horizon_days,
                context_length=min(len(prices), 512),
                patch_size="auto",
                num_samples=20,
            )

            # Run prediction
            forecasts = pipeline.forecast(ts_df)

            # Extract samples and compute quantiles
            if hasattr(forecasts, "samples"):
                samples = forecasts.samples  # (num_samples, horizon_days)
            else:
                # Fallback: use mean forecast
                samples = np.array([[float(forecasts.mean[-1])] * horizon_days])

            final_step = samples[:, -1] if samples.ndim == 2 else samples
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

            # Confidence from forecast spread
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
            self.state.metadata = {
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
            }
            return result

        except Exception:
            logger.exception("Moirai prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
