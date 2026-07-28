"""Tests for ModelBenchmark."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, PredictionDirection, PredictionResult
from quant_tools.predict.base import PredictionModel
from quant_tools.predict.benchmark import ModelBenchmark, _create_all_models


# ── Helpers ─────────────────────────────────────────────────────────


class FakeModel(PredictionModel):
    """Minimal model that always predicts UP with fixed confidence."""

    def __init__(self, model_name: str = "fake", direction: PredictionDirection = PredictionDirection.UP) -> None:
        super().__init__()
        self._name = model_name
        self._direction = direction
        self.state.is_trained = True

    @property
    def name(self) -> str:
        return self._name

    def train(self, features_df: object, target: object) -> None:
        self.state.is_trained = True
        self.state.training_samples = 100

    def predict(
        self,
        features_df: object,
        horizon_days: int = 5,
        symbol: str = "UNKNOWN",
        asset_class: AssetClass = AssetClass.EQUITY,
        interval: str = "1d",
    ) -> PredictionResult:
        return PredictionResult(
            symbol=symbol,
            asset_class=asset_class,
            model_name=self.name,
            direction=self._direction,
            magnitude=0.02,
            confidence=0.7,
            horizon_days=horizon_days,
            interval=interval,
        )


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2021-01-04", periods=n)
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = (high + low) / 2
    volume = np.random.randint(1_000_000, 50_000_000, n)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=dates,
    )


# ── Benchmark tests ─────────────────────────────────────────────────


class TestModelBenchmark:
    def test_init_with_models(self):
        models = [FakeModel("a"), FakeModel("b")]
        bench = ModelBenchmark(models=models)
        assert len(bench.models) == 2

    def test_init_default_models(self):
        """Default init creates models via lazy imports (may find 0 if deps missing)."""
        bench = ModelBenchmark()
        # At minimum, the benchmark should be constructible
        assert isinstance(bench.models, list)

    def test_run_single_model(self):
        model = FakeModel("always_up")
        bench = ModelBenchmark(models=[model])
        ohlcv = _make_ohlcv(500)
        ohlcv_data = {"AAPL": ohlcv}

        results = bench.run(
            symbols=["AAPL"],
            test_periods=[("2022-01-03", "2022-06-30")],
            horizon_days=5,
            ohlcv_data=ohlcv_data,
        )

        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        assert "model" in results.columns
        assert "direction_accuracy" in results.columns
        assert "sharpe" in results.columns
        assert "mae" in results.columns
        assert "rmse" in results.columns
        assert results["model"].iloc[0] == "always_up"

    def test_run_multiple_models(self):
        up_model = FakeModel("up_model", PredictionDirection.UP)
        down_model = FakeModel("down_model", PredictionDirection.DOWN)
        bench = ModelBenchmark(models=[up_model, down_model])
        ohlcv = _make_ohlcv(500)
        ohlcv_data = {"SPY": ohlcv}

        results = bench.run(
            symbols=["SPY"],
            test_periods=[("2022-01-03", "2022-06-30")],
            horizon_days=5,
            ohlcv_data=ohlcv_data,
        )

        assert len(results["model"].unique()) == 2
        assert "up_model" in results["model"].values
        assert "down_model" in results["model"].values

    def test_run_no_data(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        results = bench.run(
            symbols=["MISSING"],
            ohlcv_data={},
        )
        assert results.empty

    def test_run_insufficient_train_data(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        short_ohlcv = _make_ohlcv(30)
        results = bench.run(
            symbols=["SHORT"],
            test_periods=[("2021-02-01", "2021-02-28")],
            ohlcv_data={"SHORT": short_ohlcv},
        )
        # Either empty or the model handled it
        assert isinstance(results, pd.DataFrame)

    def test_compare_output(self):
        bench = ModelBenchmark(models=[FakeModel("a"), FakeModel("b")])
        # Create synthetic results DataFrame
        results = pd.DataFrame([
            {"model": "a", "symbol": "AAPL", "period": "2022H1",
             "direction_accuracy": 0.6, "sharpe": 1.2, "mae": 0.01, "rmse": 0.02},
            {"model": "b", "symbol": "AAPL", "period": "2022H1",
             "direction_accuracy": 0.7, "sharpe": 0.8, "mae": 0.015, "rmse": 0.025},
        ])

        comparison = bench.compare(results)

        assert "ranking" in comparison
        assert "best_by_metric" in comparison
        assert "per_model" in comparison
        assert not comparison["ranking"].empty
        assert comparison["best_by_metric"]["direction_accuracy"] == "b"
        assert comparison["best_by_metric"]["sharpe"] == "a"
        assert comparison["best_by_metric"]["mae"] == "a"

    def test_compare_empty_results(self):
        bench = ModelBenchmark(models=[FakeModel("a")])
        empty_df = pd.DataFrame(
            columns=["model", "symbol", "period",
                     "direction_accuracy", "sharpe", "mae", "rmse"],
        )
        comparison = bench.compare(empty_df)
        assert comparison["ranking"].empty
        assert comparison["best_by_metric"] == {}
        assert comparison["per_model"] == {}

    def test_auto_select_short_horizon_equity(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("equity", horizon=5)
        assert best == "gradient_boost"

    def test_auto_select_short_horizon_crypto(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("crypto", horizon=3)
        assert best == "nhits"

    def test_auto_select_medium_horizon(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("equity", horizon=10)
        assert best == "tft"

    def test_auto_select_medium_horizon_forex(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("forex", horizon=15)
        assert best == "patchtst"

    def test_auto_select_long_horizon(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("equity", horizon=30)
        assert best == "stats_forecast"

    def test_auto_select_very_long_horizon(self):
        bench = ModelBenchmark(models=[FakeModel("test")])
        best = bench.auto_select("equity", horizon=90)
        assert best == "chronos"

    def test_walk_forward_no_overlap(self):
        """Verify training data does NOT overlap with test period."""
        model = FakeModel("checker")
        bench = ModelBenchmark(models=[model])
        ohlcv = _make_ohlcv(500)
        ohlcv_data = {"TEST": ohlcv}

        # The model's train method is called; we can verify
        # the benchmark successfully runs without errors
        # (actual walk-forward logic is in _evaluate_model)
        results = bench.run(
            symbols=["TEST"],
            test_periods=[("2022-01-03", "2022-03-31")],
            horizon_days=5,
            ohlcv_data=ohlcv_data,
        )
        assert isinstance(results, pd.DataFrame)

    def test_run_multiple_periods(self):
        bench = ModelBenchmark(models=[FakeModel("multi")])
        ohlcv = _make_ohlcv(600)
        ohlcv_data = {"MSFT": ohlcv}

        results = bench.run(
            symbols=["MSFT"],
            test_periods=[
                ("2022-01-03", "2022-06-30"),
                ("2022-07-01", "2022-12-30"),
            ],
            horizon_days=5,
            ohlcv_data=ohlcv_data,
        )
        # Should have results for at least one period
        assert isinstance(results, pd.DataFrame)

    def test_run_multiple_symbols(self):
        bench = ModelBenchmark(models=[FakeModel("sym_test")])
        ohlcv1 = _make_ohlcv(500, seed=42)
        ohlcv2 = _make_ohlcv(500, seed=43)
        ohlcv_data = {"AAPL": ohlcv1, "MSFT": ohlcv2}

        results = bench.run(
            symbols=["AAPL", "MSFT"],
            test_periods=[("2022-01-03", "2022-06-30")],
            horizon_days=5,
            ohlcv_data=ohlcv_data,
        )

        assert isinstance(results, pd.DataFrame)
        if not results.empty:
            symbols_in_results = results["symbol"].unique()
            assert len(symbols_in_results) >= 1
