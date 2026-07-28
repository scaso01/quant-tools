"""Facebook Prophet model for time-series forecasting."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger(__name__)

# Interval -> Prophet/pandas frequency string
_INTERVAL_TO_PROPHET_FREQ: dict[str, str] = {
    "1d": "B",
    "1h": "h",
    "4h": "4h",
    "15m": "15min",
}


class ProphetModel(PredictionModel):
    """Prophet-based trend + seasonality forecast model."""

    def __init__(self) -> None:
        super().__init__()
        self._close_prices: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "prophet"

    def train(self, features_df: object, target: object) -> None:
        """Store close prices for Prophet forecasting."""
        try:
            df = features_df
            if hasattr(df, "columns") and "close" in df.columns:
                self._close_prices = np.asarray(df["close"]).ravel()
            elif hasattr(df, "columns") and len(df.columns) > 0:
                self._close_prices = np.asarray(df.iloc[:, 0]).ravel()
            else:
                self._close_prices = np.asarray(df).ravel()

            if len(self._close_prices) < 30:
                logger.warning("Insufficient data for Prophet (%d rows)", len(self._close_prices))
                return

            self.state.is_trained = True
            self.state.last_trained = datetime.now()
            self.state.training_samples = len(self._close_prices)
        except Exception:
            logger.exception("Prophet training failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Forecast using Facebook Prophet."""
        if not self.state.is_trained or self._close_prices is None:
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
            # Suppress cmdstanpy and prophet logs
            _prev_cmdstan = os.environ.get("CMDSTAN_VERBOSE", None)
            os.environ["CMDSTAN_VERBOSE"] = "0"
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            logging.getLogger("prophet").setLevel(logging.WARNING)

            from prophet import Prophet
        except ImportError:
            self.state.is_trained = False
            logger.warning("prophet not installed")
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

            prices = self._close_prices
            n = len(prices)
            prophet_freq = _INTERVAL_TO_PROPHET_FREQ.get(interval, "B")
            is_intraday = interval != "1d"
            ds = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=prophet_freq)

            train_df = pd.DataFrame({"ds": ds, "y": prices.astype(float)})

            m = Prophet(
                daily_seasonality=is_intraday,
                weekly_seasonality=not is_intraday,
                yearly_seasonality=not is_intraday,
            )
            m.fit(train_df)

            future = m.make_future_dataframe(periods=horizon_days, freq=prophet_freq)
            forecast = m.predict(future)

            # Get forecast for the horizon window
            future_rows = forecast.tail(horizon_days)
            yhat_end = float(future_rows["yhat"].values[-1])
            current = float(prices[-1])

            pct_change = (yhat_end - current) / current if current != 0 else 0.0

            # Direction from trend slope
            if pct_change > 0.001:
                direction = PredictionDirection.UP
            elif pct_change < -0.001:
                direction = PredictionDirection.DOWN
            else:
                direction = PredictionDirection.NEUTRAL

            # Confidence from prediction interval width
            yhat_upper = float(future_rows["yhat_upper"].values[-1])
            yhat_lower = float(future_rows["yhat_lower"].values[-1])
            interval_width = (yhat_upper - yhat_lower) / current if current != 0 else 1.0
            confidence = max(0.2, min(0.9, 1.0 - interval_width))

            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=direction,
                magnitude=abs(pct_change),
                confidence=confidence,
                horizon_days=horizon_days,
                interval=interval,
            )
        except Exception:
            logger.exception("Prophet prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
        finally:
            # Restore cmdstan env var
            if _prev_cmdstan is not None:
                os.environ["CMDSTAN_VERBOSE"] = _prev_cmdstan
            elif "CMDSTAN_VERBOSE" in os.environ:
                del os.environ["CMDSTAN_VERBOSE"]
