"""Google TimesFM foundation model for zero-shot time-series forecasting."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger("quant_tools")

_MIN_DATA_POINTS = 50


class TimesFMModel(PredictionModel):
    """TimesFM (Google) pretrained foundation model for zero-shot forecasting.

    This model does NOT train on local data. The ``train()`` method loads the
    pretrained checkpoint from Hugging Face.  ``predict()`` runs zero-shot
    inference on close prices and returns probabilistic quantile forecasts in
    ``PredictionResult.metadata``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: object | None = None

    @property
    def name(self) -> str:
        return "timesfm"

    def train(self, features_df: object, target: object) -> None:
        """Load TimesFM checkpoint (no actual training)."""
        try:
            import timesfm

            self._model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    per_core_batch_size=32,
                    horizon_len=128,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
                ),
            )
            self.state.is_trained = True
            self.state.last_trained = datetime.now()
        except ImportError:
            logger.warning(
                "timesfm not installed. pip install quant-tools[foundation]"
            )
        except Exception:
            logger.exception("TimesFM checkpoint loading failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Zero-shot forecast using TimesFM foundation model."""
        if not self.state.is_trained or self._model is None:
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

            # Frequency mapping for TimesFM
            freq_map = {"1d": 0, "1h": 0, "4h": 0, "15m": 0}
            frequency_input = [freq_map.get(interval, 0)]

            forecast_input = [prices.astype(float)]
            point_forecast, experimental_quantile_forecast = self._model.forecast(
                forecast_input,
                freq=frequency_input,
            )

            # point_forecast shape: (1, horizon_len) — take first horizon_days
            forecast_values = point_forecast[0][:horizon_days]
            forecast_end = float(forecast_values[-1])
            current = float(prices[-1])

            # Quantiles from experimental output if available
            if experimental_quantile_forecast is not None and len(experimental_quantile_forecast) > 0:
                q_forecast = experimental_quantile_forecast[0][:horizon_days]
                # Shape: (horizon_days, num_quantiles) — typically 10/25/50/75/90
                final_q = q_forecast[-1]
                if len(final_q) >= 5:
                    q10, q25, q50, q75, q90 = (
                        float(final_q[0]),
                        float(final_q[1]),
                        float(final_q[2]),
                        float(final_q[3]),
                        float(final_q[4]),
                    )
                else:
                    q10 = q25 = q50 = q75 = q90 = forecast_end
            else:
                q10 = q25 = q50 = q75 = q90 = forecast_end

            # Direction from median/point forecast vs current price
            pct_change = (forecast_end - current) / current if current != 0 else 0.0

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
            logger.exception("TimesFM prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
