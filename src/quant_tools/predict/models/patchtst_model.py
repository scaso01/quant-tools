"""PatchTST model using neuralforecast."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger("quant_tools")

_MIN_DATA_POINTS = 50

# Interval -> neuralforecast frequency string
_INTERVAL_TO_NF_FREQ: dict[str, str] = {
    "1d": "B",
    "1h": "h",
    "4h": "4h",
    "15m": "15min",
}

# Interval -> bars_per_day multiplier for input_size scaling
_INTERVAL_TO_BARS_PER_DAY: dict[str, int] = {
    "1d": 1,
    "1h": 24,
    "4h": 6,
    "15m": 96,
}

# Interval -> season_length
_INTERVAL_TO_SEASON: dict[str, int] = {
    "1d": 5,
    "1h": 24,
    "4h": 6,
    "15m": 96,
}


class PatchTSTModel(PredictionModel):
    """PatchTST (Patch Time Series Transformer) for time-series forecasting."""

    def __init__(self) -> None:
        super().__init__()
        self._close_prices: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "patchtst"

    def train(self, features_df: object, target: object) -> None:
        """Store close prices for PatchTST forecasting."""
        try:
            df = features_df
            if hasattr(df, "columns") and "close" in df.columns:
                self._close_prices = np.asarray(df["close"]).ravel()
            elif hasattr(df, "columns") and len(df.columns) > 0:
                self._close_prices = np.asarray(df.iloc[:, 0]).ravel()
            else:
                self._close_prices = np.asarray(df).ravel()

            if len(self._close_prices) < _MIN_DATA_POINTS:
                logger.warning(
                    "PatchTST requires at least %d data points, got %d",
                    _MIN_DATA_POINTS,
                    len(self._close_prices),
                )
                return

            self.state.is_trained = True
            self.state.last_trained = datetime.now()
            self.state.training_samples = len(self._close_prices)
        except Exception:
            logger.exception("PatchTST training failed")

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        """Forecast using PatchTST."""
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

        if len(self._close_prices) < _MIN_DATA_POINTS:
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
            from neuralforecast import NeuralForecast
            from neuralforecast.models import PatchTST
        except ImportError:
            self.state.is_trained = False
            logger.warning("neuralforecast not installed")
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
            nf_freq = _INTERVAL_TO_NF_FREQ.get(interval, "B")
            bars_per_day = _INTERVAL_TO_BARS_PER_DAY.get(interval, 1)
            input_size = 30 * bars_per_day
            ds = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=nf_freq)

            train_df = pd.DataFrame({
                "unique_id": ["asset"] * n,
                "ds": ds,
                "y": prices.astype(float),
            })

            model = PatchTST(
                h=horizon_days,
                input_size=input_size,
                hidden_size=64,
                n_heads=4,
                patch_len=16,
                max_steps=100,
                scaler_type="standard",
            )
            nf = NeuralForecast(models=[model], freq=nf_freq)
            nf.fit(train_df)
            forecast = nf.predict()

            yhat = forecast["PatchTST"].values
            forecast_end = float(yhat[-1])
            current = float(prices[-1])

            pct_change = (forecast_end - current) / current if current != 0 else 0.0

            if pct_change > 0.001:
                direction = PredictionDirection.UP
            elif pct_change < -0.001:
                direction = PredictionDirection.DOWN
            else:
                direction = PredictionDirection.NEUTRAL

            # Confidence based on training data size and forecast stability
            stability = 1.0 - min(1.0, np.std(yhat) / abs(current) if current != 0 else 1.0)
            data_factor = min(1.0, len(prices) / 500)
            confidence = max(0.2, min(0.85, 0.5 * stability + 0.5 * data_factor))

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
            logger.exception("PatchTST prediction failed")
            return PredictionResult(
                symbol=symbol,
                asset_class=asset_class,
                model_name=self.name,
                direction=PredictionDirection.NEUTRAL,
                confidence=0.3,
                horizon_days=horizon_days,
                interval=interval,
            )
