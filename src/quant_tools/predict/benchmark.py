"""Benchmark prediction models on standardized test periods."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel

logger = logging.getLogger("quant_tools")

# Default test periods (start, end) for walk-forward evaluation
_DEFAULT_PERIODS: list[tuple[str, str]] = [
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
]


def _create_all_models() -> list[PredictionModel]:
    """Lazy-create all available prediction models."""
    models: list[PredictionModel] = []

    model_imports: list[tuple[str, str]] = [
        ("quant_tools.predict.models.gradient_boost", "GradientBoostModel"),
        ("quant_tools.predict.models.stats_forecast", "StatsForecastModel"),
        ("quant_tools.predict.models.neural_forecast", "NeuralForecastModel"),
        ("quant_tools.predict.models.prophet_model", "ProphetModel"),
        ("quant_tools.predict.models.garch", "GARCHModel"),
        ("quant_tools.predict.models.tft_model", "TFTModel"),
        ("quant_tools.predict.models.patchtst_model", "PatchTSTModel"),
        ("quant_tools.predict.models.nhits_model", "NHiTSModel"),
        ("quant_tools.predict.models.timesnet_model", "TimesNetModel"),
        ("quant_tools.predict.models.chronos_model", "ChronosModel"),
        ("quant_tools.predict.models.timesfm_model", "TimesFMModel"),
        ("quant_tools.predict.models.moirai_model", "MoiraiModel"),
    ]

    for module_path, class_name in model_imports:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            models.append(cls())
        except Exception as exc:
            logger.debug("Could not load %s: %s", class_name, exc)

    return models


class ModelBenchmark:
    """Benchmark prediction models on standardized test periods.

    Supports walk-forward evaluation to avoid look-ahead bias: each test
    period is trained only on data that precedes it.
    """

    def __init__(self, models: list[PredictionModel] | None = None) -> None:
        """Initialize with specific models or all available models.

        Args:
            models: List of PredictionModel instances. If ``None``,
                creates all available models via lazy imports.
        """
        if models is None:
            self.models = _create_all_models()
        else:
            self.models = list(models)

    def run(
        self,
        symbols: list[str],
        test_periods: list[tuple[str, str]] | None = None,
        horizon_days: int = 5,
        ohlcv_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Walk-forward evaluation of models.

        For each symbol and test period, the model is trained on data before
        the test period start, then predictions are evaluated against actual
        outcomes within the test period.

        Args:
            symbols: Ticker symbols to evaluate.
            test_periods: List of ``(start_date, end_date)`` strings.
                Defaults to semi-annual periods in 2022-2023.
            horizon_days: Prediction horizon in days.
            ohlcv_data: Pre-loaded OHLCV DataFrames keyed by symbol.
                If ``None``, data is fetched via the provider registry
                (requires network access).

        Returns:
            DataFrame with columns: model, symbol, period,
            direction_accuracy, sharpe, mae, rmse.
        """
        if test_periods is None:
            test_periods = _DEFAULT_PERIODS

        rows: list[dict] = []

        for symbol in symbols:
            df = self._get_ohlcv(symbol, ohlcv_data)
            if df is None or df.empty:
                logger.warning("No data for %s, skipping", symbol)
                continue

            for period_start, period_end in test_periods:
                period_label = f"{period_start}/{period_end}"
                start_dt = pd.Timestamp(period_start)
                end_dt = pd.Timestamp(period_end)

                # Training data: everything before the test period
                train_mask = df.index < start_dt
                # Test data: within the test period
                test_mask = (df.index >= start_dt) & (df.index <= end_dt)

                train_df = df[train_mask]
                test_df = df[test_mask]

                if len(train_df) < 60 or len(test_df) < horizon_days:
                    continue

                for model in self.models:
                    try:
                        metrics = self._evaluate_model(
                            model, train_df, test_df, horizon_days,
                        )
                        rows.append({
                            "model": model.name,
                            "symbol": symbol,
                            "period": period_label,
                            **metrics,
                        })
                    except Exception as exc:
                        logger.debug(
                            "Benchmark %s/%s/%s failed: %s",
                            model.name, symbol, period_label, exc,
                        )

        if not rows:
            return pd.DataFrame(
                columns=["model", "symbol", "period", "direction_accuracy",
                         "sharpe", "mae", "rmse"],
            )

        return pd.DataFrame(rows)

    def _evaluate_model(
        self,
        model: PredictionModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        horizon_days: int,
    ) -> dict:
        """Evaluate a single model on one period.

        Returns dict with direction_accuracy, sharpe, mae, rmse.
        """
        from quant_tools.predict.features import FeatureEngineer

        engineer = FeatureEngineer()

        # Train
        feature_set = engineer.build_features(train_df, horizon_days=horizon_days)
        if model.name in {"stats_forecast", "neural_forecast", "prophet", "garch",
                          "tft", "patchtst", "nhits", "timesnet"}:
            train_features = feature_set.features_df.copy()
            if feature_set.close_prices is not None:
                train_features["close"] = feature_set.close_prices.values
        else:
            train_features = feature_set.features_df

        model.train(train_features, feature_set.target)

        # Walk-forward predictions on test data
        close_prices = test_df["Close"].values
        actual_returns = []
        predicted_directions = []
        predicted_magnitudes = []

        step = max(1, horizon_days)
        for i in range(0, len(close_prices) - horizon_days, step):
            # Use all data up to prediction point
            context_end = test_df.index[i]
            context_df = pd.concat([train_df, test_df.loc[:context_end]])

            feat = engineer.build_features(context_df, horizon_days=horizon_days)
            if model.name in {"stats_forecast", "neural_forecast", "prophet", "garch",
                              "tft", "patchtst", "nhits", "timesnet"}:
                pred_features = feat.features_df.copy()
                if feat.close_prices is not None:
                    pred_features["close"] = feat.close_prices.values
            else:
                pred_features = feat.features_df

            result = model.predict(pred_features, horizon_days=horizon_days)

            actual_ret = (close_prices[i + horizon_days] - close_prices[i]) / close_prices[i]
            actual_returns.append(actual_ret)
            predicted_directions.append(result.direction)
            predicted_magnitudes.append(result.magnitude if result.magnitude is not None else 0.0)

        if not actual_returns:
            return {
                "direction_accuracy": 0.0,
                "sharpe": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
            }

        actual_arr = np.array(actual_returns)
        pred_mag = np.array(predicted_magnitudes)

        # Direction accuracy
        actual_dirs = []
        for r in actual_returns:
            if r > 0.001:
                actual_dirs.append(PredictionDirection.UP)
            elif r < -0.001:
                actual_dirs.append(PredictionDirection.DOWN)
            else:
                actual_dirs.append(PredictionDirection.NEUTRAL)

        correct = sum(1 for a, p in zip(actual_dirs, predicted_directions) if a == p)
        direction_accuracy = correct / len(actual_dirs) if actual_dirs else 0.0

        # Strategy returns (long if UP, short if DOWN, flat if NEUTRAL)
        strategy_returns = []
        for ret, pred_dir in zip(actual_returns, predicted_directions):
            if pred_dir == PredictionDirection.UP:
                strategy_returns.append(ret)
            elif pred_dir == PredictionDirection.DOWN:
                strategy_returns.append(-ret)
            else:
                strategy_returns.append(0.0)

        strat_arr = np.array(strategy_returns)
        sharpe = (
            float(np.mean(strat_arr) / (np.std(strat_arr) + 1e-10) * np.sqrt(252))
            if len(strat_arr) > 1
            else 0.0
        )

        # MAE and RMSE on magnitude predictions
        mae = float(np.mean(np.abs(pred_mag - np.abs(actual_arr))))
        rmse = float(np.sqrt(np.mean((pred_mag - np.abs(actual_arr)) ** 2)))

        return {
            "direction_accuracy": round(direction_accuracy, 4),
            "sharpe": round(sharpe, 4),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
        }

    @staticmethod
    def _get_ohlcv(
        symbol: str,
        ohlcv_data: dict[str, pd.DataFrame] | None,
    ) -> pd.DataFrame | None:
        """Get OHLCV data from cache or provider."""
        if ohlcv_data is not None and symbol in ohlcv_data:
            return ohlcv_data[symbol]

        try:
            from quant_tools.data.providers.registry import ProviderRegistry

            registry = ProviderRegistry()
            asset_class = registry.detect_asset_class(symbol)
            provider = registry.get_provider(asset_class)
            return provider.fetch_ohlcv(symbol, period="5y")
        except Exception as exc:
            logger.warning("Could not fetch data for %s: %s", symbol, exc)
            return None

    def compare(self, results: pd.DataFrame) -> dict:
        """Aggregate comparison of benchmark results.

        Args:
            results: DataFrame from ``run()`` with columns model, symbol,
                period, direction_accuracy, sharpe, mae, rmse.

        Returns:
            Dict with keys:
            - ``ranking``: DataFrame of models ranked by composite score
            - ``best_by_metric``: dict of metric -> best model name
            - ``per_model``: dict of model -> aggregated metrics
        """
        if results.empty:
            return {
                "ranking": pd.DataFrame(),
                "best_by_metric": {},
                "per_model": {},
            }

        metrics = ["direction_accuracy", "sharpe", "mae", "rmse"]
        agg = results.groupby("model")[metrics].mean()

        # For MAE and RMSE, lower is better -> invert for ranking
        rank_df = agg.copy()
        rank_df["mae_inv"] = 1.0 / (rank_df["mae"] + 1e-10)
        rank_df["rmse_inv"] = 1.0 / (rank_df["rmse"] + 1e-10)

        # Composite score: normalize each metric to [0,1] and average
        for col in ["direction_accuracy", "sharpe", "mae_inv", "rmse_inv"]:
            col_min = rank_df[col].min()
            col_max = rank_df[col].max()
            col_range = col_max - col_min
            rank_df[f"{col}_norm"] = (
                (rank_df[col] - col_min) / col_range if col_range > 0 else 0.5
            )

        rank_df["composite"] = (
            rank_df["direction_accuracy_norm"]
            + rank_df["sharpe_norm"]
            + rank_df["mae_inv_norm"]
            + rank_df["rmse_inv_norm"]
        ) / 4.0

        ranking = rank_df[["composite"]].sort_values("composite", ascending=False)
        ranking = ranking.join(agg)

        best_by_metric = {
            "direction_accuracy": agg["direction_accuracy"].idxmax() if not agg.empty else "",
            "sharpe": agg["sharpe"].idxmax() if not agg.empty else "",
            "mae": agg["mae"].idxmin() if not agg.empty else "",
            "rmse": agg["rmse"].idxmin() if not agg.empty else "",
        }

        per_model = {}
        for model_name in agg.index:
            per_model[model_name] = agg.loc[model_name].to_dict()

        return {
            "ranking": ranking,
            "best_by_metric": best_by_metric,
            "per_model": per_model,
        }

    def auto_select(self, asset_class: str, horizon: int) -> str:
        """Return name of best model for given asset class and horizon.

        This is a heuristic-based selection that returns reasonable defaults
        based on model characteristics. For data-driven selection, run
        ``run()`` + ``compare()`` on historical data first.

        Args:
            asset_class: Asset class string (e.g. "equity", "crypto").
            horizon: Prediction horizon in days.

        Returns:
            Model name string (e.g. "gradient_boost", "tft").
        """
        # Short-horizon: gradient boosting excels
        if horizon <= 5:
            if asset_class == "crypto":
                return "nhits"  # NHiTS handles volatile crypto well
            return "gradient_boost"

        # Medium-horizon: transformers shine
        if horizon <= 20:
            if asset_class == "forex":
                return "patchtst"  # PatchTST captures FX patterns
            return "tft"  # TFT is strong general-purpose

        # Long-horizon: statistical/foundation models
        if horizon <= 60:
            return "stats_forecast"  # AutoARIMA for longer horizons

        # Very long: foundation models with pre-trained knowledge
        return "chronos"
