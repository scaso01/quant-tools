"""Weighted ensemble combiner for multi-model predictions."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

import numpy as np

from quant_tools.config import cfg
from quant_tools.errors import PredictionError
from quant_tools.models import (
    AssetClass,
    ConsensusSignal,
    EnsemblePrediction,
    PredictionDirection,
    PredictionResult,
)

log = logging.getLogger("quant_tools")

# Minimum weight floor to prevent model death
MIN_WEIGHT = 0.05


class PredictionEnsemble:
    """Combines predictions from multiple models via weighted voting."""

    def __init__(
        self,
        models: list | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize ensemble with models and optional weights.

        Args:
            models: List of PredictionModel instances. If None, creates all 5 default models.
            weights: Dict of model_name -> weight. If None, uses equal weights.
        """
        self._lock = threading.Lock()

        if models is None:
            models = self._create_default_models()

        self.models = models

        if weights is None:
            n = len(self.models)
            self._weights = {m.name: 1.0 / n for m in self.models}
        else:
            self._weights = dict(weights)
            self._enforce_floor()

    @staticmethod
    def _create_default_models() -> list:
        """Lazy-create all 5 default models."""
        models: list = []
        try:
            from quant_tools.predict.models.gradient_boost import GradientBoostModel
            models.append(GradientBoostModel())
        except Exception as exc:
            log.warning("Could not load GradientBoostModel: %s", exc)
        try:
            from quant_tools.predict.models.stats_forecast import StatsForecastModel
            models.append(StatsForecastModel())
        except Exception as exc:
            log.warning("Could not load StatsForecastModel: %s", exc)
        try:
            from quant_tools.predict.models.neural_forecast import NeuralForecastModel
            models.append(NeuralForecastModel())
        except Exception as exc:
            log.warning("Could not load NeuralForecastModel: %s", exc)
        try:
            from quant_tools.predict.models.prophet_model import ProphetModel
            models.append(ProphetModel())
        except Exception as exc:
            log.warning("Could not load ProphetModel: %s", exc)
        try:
            from quant_tools.predict.models.garch import GARCHModel
            models.append(GARCHModel())
        except Exception as exc:
            log.warning("Could not load GARCHModel: %s", exc)

        # NeuralForecast expansion (optional — requires neuralforecast 1.7+)
        try:
            from quant_tools.predict.models.tft_model import TFTModel
            models.append(TFTModel())
        except Exception:
            pass
        try:
            from quant_tools.predict.models.patchtst_model import PatchTSTModel
            models.append(PatchTSTModel())
        except Exception:
            pass
        try:
            from quant_tools.predict.models.nhits_model import NHiTSModel
            models.append(NHiTSModel())
        except Exception:
            pass
        try:
            from quant_tools.predict.models.timesnet_model import TimesNetModel
            models.append(TimesNetModel())
        except Exception:
            pass

        # Foundation models (optional — requires [foundation] extra)
        try:
            from quant_tools.predict.models.chronos_model import ChronosModel
            models.append(ChronosModel())
        except Exception:
            pass
        try:
            from quant_tools.predict.models.timesfm_model import TimesFMModel
            models.append(TimesFMModel())
        except Exception:
            pass
        try:
            from quant_tools.predict.models.moirai_model import MoiraiModel
            models.append(MoiraiModel())
        except Exception:
            pass

        if not models:
            raise PredictionError("No prediction models could be loaded")
        return models

    def _enforce_floor(self) -> None:
        """Ensure all weights >= MIN_WEIGHT and sum to 1."""
        for name in self._weights:
            self._weights[name] = max(self._weights[name], MIN_WEIGHT)
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total

    def predict(
        self,
        ohlcv_df: object,
        symbol: str,
        asset_class: AssetClass,
        horizon_days: int | None = None,
        hours: int | None = None,
        interval: str = "1d",
    ) -> EnsemblePrediction:
        """Run all models and combine predictions.

        Args:
            ohlcv_df: OHLCV DataFrame (Open, High, Low, Close, Volume).
            symbol: Ticker/pair symbol.
            asset_class: Asset class.
            horizon_days: Prediction horizon in days. Defaults to config value.
            hours: Intraday prediction horizon in hours.  When set, the target
                date/time is ``now + hours`` and *horizon_days* is derived
                automatically if not provided (``max(1, hours // 24)``).
            interval: Bar interval for the OHLCV data (e.g. ``"1d"``,
                ``"1h"``, ``"15m"``).  Passed through to feature engineering
                and individual model ``predict()`` calls.

        Returns:
            EnsemblePrediction with consensus signal and individual model results.
        """
        from quant_tools.predict.features import FeatureEngineer, _bars_per_day

        # Resolve horizon and target date
        if hours is not None:
            target_dt = datetime.now() + timedelta(hours=hours)
            target_date = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            if horizon_days is None:
                horizon_days = max(1, hours // 24)
        else:
            if horizon_days is None:
                horizon_days = cfg.prediction_horizon_days
            target_date = (
                datetime.now() + timedelta(days=horizon_days)
            ).strftime("%Y-%m-%d")

        # Compute horizon_bars for intraday intervals
        horizon_bars: int | None = None
        if hours is not None:
            bpd = _bars_per_day(interval)
            horizon_bars = max(1, int(hours * bpd / 24))

        # Build features
        engineer = FeatureEngineer()
        feature_set = engineer.build_features(
            ohlcv_df, symbol, asset_class, horizon_days,
            interval=interval, horizon_bars=horizon_bars,
        )

        # Build features_df with close prices for time-series models
        _TS_MODELS = {"stats_forecast", "neural_forecast", "prophet", "garch"}
        features_with_close = feature_set.features_df.copy()
        if feature_set.close_prices is not None:
            features_with_close["close"] = feature_set.close_prices.values

        # Run each model
        model_predictions: list[PredictionResult] = []
        garch_confidence: float | None = None

        for model in self.models:
            if not model.supports_asset_class(asset_class):
                continue
            try:
                train_df = features_with_close if model.name in _TS_MODELS else feature_set.features_df
                model.train(train_df, feature_set.target)
                pred = model.predict(train_df, horizon_days, interval=interval)
                pred.symbol = symbol
                pred.asset_class = asset_class
                pred.target_date = target_date
                pred.horizon_days = horizon_days
                pred.interval = interval
                model_predictions.append(pred)

                if model.name == "garch":
                    garch_confidence = pred.confidence
            except Exception as exc:
                log.warning("Model %s failed for %s: %s", model.name, symbol, exc)

        if not model_predictions:
            raise PredictionError(f"All models failed for {symbol}")

        # Refuse to emit fallback-only output as if it were a real prediction
        trained_count = sum(1 for m in self.models if m.state.is_trained)
        if trained_count == 0:
            raise PredictionError(
                f"No trained models for {symbol}: every model returned an untrained "
                "fallback (are the ML packages installed?). "
                "Install them with: pip install 'quant-tools[ml]' (requires Python <= 3.13)."
            )

        result = self._combine(
            model_predictions,
            symbol,
            asset_class,
            horizon_days,
            target_date,
            garch_confidence,
            interval=interval,
        )

        # Stamp horizon_hours for intraday predictions
        if hours is not None:
            result.horizon_hours = hours
            for mp in result.model_predictions:
                mp.horizon_hours = hours

        return result

    def _combine(
        self,
        predictions: list[PredictionResult],
        symbol: str,
        asset_class: AssetClass,
        horizon_days: int,
        target_date: str,
        garch_confidence: float | None,
        interval: str = "1d",
    ) -> EnsemblePrediction:
        """Weighted voting combination of model predictions.

        Steps:
        1. Weighted vote on direction (exclude GARCH which is always NEUTRAL)
        2. Weighted average of magnitudes from agreeing models
        3. Confidence = weighted fraction agreeing * avg individual confidence
        4. GARCH volatility adjusts final confidence
        5. Map to ConsensusSignal
        """
        # Separate direction-voting models from GARCH
        direction_preds = [p for p in predictions if p.model_name != "garch"]

        if not direction_preds:
            return EnsemblePrediction(
                symbol=symbol,
                asset_class=asset_class,
                consensus=ConsensusSignal.HOLD,
                confidence=0.3,
                direction=PredictionDirection.NEUTRAL,
                horizon_days=horizon_days,
                target_date=target_date,
                interval=interval,
                model_predictions=predictions,
            )

        # Weighted direction votes
        up_weight = 0.0
        down_weight = 0.0
        neutral_weight = 0.0

        for pred in direction_preds:
            w = self._weights.get(pred.model_name, 1.0 / len(direction_preds))
            if pred.direction == PredictionDirection.UP:
                up_weight += w
            elif pred.direction == PredictionDirection.DOWN:
                down_weight += w
            else:
                neutral_weight += w

        total_directional = up_weight + down_weight + neutral_weight

        # Determine consensus direction
        if up_weight >= down_weight and up_weight >= neutral_weight:
            direction = PredictionDirection.UP
            agreement_weight = up_weight
        elif down_weight >= up_weight and down_weight >= neutral_weight:
            direction = PredictionDirection.DOWN
            agreement_weight = down_weight
        else:
            direction = PredictionDirection.NEUTRAL
            agreement_weight = neutral_weight

        # Weighted magnitude from agreeing models
        agreeing = [
            p
            for p in direction_preds
            if p.direction == direction and p.magnitude is not None
        ]
        if agreeing:
            magnitudes = []
            weights_for_mag = []
            for p in agreeing:
                magnitudes.append(p.magnitude)
                weights_for_mag.append(self._weights.get(p.model_name, 0.1))
            w_arr = np.array(weights_for_mag)
            w_arr = w_arr / (w_arr.sum() + 1e-10)
            magnitude = float(np.average(magnitudes, weights=w_arr))
        else:
            magnitude = None

        # Confidence calculation
        agreement_fraction = agreement_weight / (total_directional + 1e-10)
        avg_conf = float(np.mean([p.confidence for p in direction_preds]))
        confidence = agreement_fraction * avg_conf

        # GARCH volatility adjustment (high vol = lower confidence)
        if garch_confidence is not None:
            confidence *= 0.5 + 0.5 * garch_confidence

        confidence = min(max(confidence, 0.0), 1.0)

        # Map to ConsensusSignal
        consensus = self._map_consensus(direction, confidence)

        return EnsemblePrediction(
            symbol=symbol,
            asset_class=asset_class,
            consensus=consensus,
            confidence=round(confidence, 4),
            direction=direction,
            magnitude=round(magnitude, 4) if magnitude is not None else None,
            horizon_days=horizon_days,
            target_date=target_date,
            interval=interval,
            model_predictions=predictions,
        )

    @staticmethod
    def _map_consensus(
        direction: PredictionDirection,
        confidence: float,
    ) -> ConsensusSignal:
        """Map direction + confidence to ConsensusSignal."""
        if direction == PredictionDirection.NEUTRAL:
            return ConsensusSignal.HOLD
        if direction == PredictionDirection.UP:
            return ConsensusSignal.STRONG_BUY if confidence > 0.7 else ConsensusSignal.BUY
        # DOWN
        return ConsensusSignal.STRONG_SELL if confidence > 0.7 else ConsensusSignal.SELL

    def update_weights(self, performance: dict[str, float]) -> None:
        """Update model weights based on tracked win rates.

        Args:
            performance: Dict of model_name -> win_rate (0-1).
        """
        with self._lock:
            for name, win_rate in performance.items():
                if name in self._weights:
                    self._weights[name] = win_rate
            # Softmax normalization
            values = np.array(list(self._weights.values()))
            exp_vals = np.exp(values - np.max(values))
            softmax = exp_vals / (exp_vals.sum() + 1e-10)
            for i, name in enumerate(self._weights):
                self._weights[name] = float(softmax[i])
            self._enforce_floor()

    def get_weights(self) -> dict[str, float]:
        """Return current model weights."""
        with self._lock:
            return dict(self._weights)
