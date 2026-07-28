"""Statistical forecasting model using AutoARIMA + AutoETS."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger(__name__)

# Interval -> statsforecast frequency string
_INTERVAL_TO_SF_FREQ: dict[str, str] = {
    "1d": "B",
    "1h": "h",
    "4h": "4h",
    "15m": "15min",
}

# Interval -> season_length for AutoARIMA
_INTERVAL_TO_SEASON: dict[str, int] = {
    "1d": 5,       # 5 business days per week
    "1h": 24,      # 24 hours per day
    "4h": 6,       # 6 four-hour bars per day
    "15m": 96,     # 96 fifteen-minute bars per day
}


class StatsForecastModel(PredictionModel):
    """AutoARIMA-based statistical forecast model."""

    def __init__(self) -> None:
        super().__init__()
        self._close_prices: np.ndarray | None = None
        self._dates: list | None = None

    @property
    def name(self) -> str:
        return "stats_forecast"

    def train(self, features_df: object, target: object) -> None:
        """Store close prices for time-series forecasting."""
        try:
            df = features_df
            if hasattr(df, "columns") and "close" in df.columns:
                self._close_prices = np.asarray(df["close"]).ravel()
            elif hasattr(df, "columns") and len(df.columns) > 0:
                self._close_prices = np.asarray(df.iloc[:, 0]).ravel()
            else:
                self._close_prices = np.asarray(df).ravel()

            if len(self._close_prices) < 30:
                logger.warning("Insufficient data for stats_forecast (%d rows)", len(self._close_prices))
                return

            self.state.is_trained = True
            self.state.last_trained = datetime.now()
            self.state.training_samples = len(self._close_prices)
        except Exception:
            logger.exception("StatsForecast training failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Forecast using AutoARIMA on close price series."""
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
            from statsforecast import StatsForecast as SF
            from statsforecast.models import AutoARIMA
        except ImportError:
            self.state.is_trained = False
            logger.warning("statsforecast not installed")
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
            sf_freq = _INTERVAL_TO_SF_FREQ.get(interval, "B")
            season_length = _INTERVAL_TO_SEASON.get(interval, 5)
            ds = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=sf_freq)

            train_df = pd.DataFrame({
                "unique_id": ["asset"] * n,
                "ds": ds,
                "y": prices,
            })

            sf = SF(models=[AutoARIMA(season_length=season_length)], freq=sf_freq)
            sf.fit(train_df)
            forecast = sf.predict(h=horizon_days, level=[90])

            yhat = forecast["AutoARIMA"].values
            forecast_end = float(yhat[-1])
            current = float(prices[-1])

            pct_change = (forecast_end - current) / current if current != 0 else 0.0

            if pct_change > 0.001:
                direction = PredictionDirection.UP
            elif pct_change < -0.001:
                direction = PredictionDirection.DOWN
            else:
                direction = PredictionDirection.NEUTRAL

            # Confidence from prediction interval width
            if "AutoARIMA-lo-90" in forecast.columns and "AutoARIMA-hi-90" in forecast.columns:
                lo = float(forecast["AutoARIMA-lo-90"].values[-1])
                hi = float(forecast["AutoARIMA-hi-90"].values[-1])
                interval_width = (hi - lo) / current if current != 0 else 1.0
                confidence = max(0.2, min(0.95, 1.0 - interval_width))
            else:
                confidence = 0.5

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
            logger.exception("StatsForecast prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
