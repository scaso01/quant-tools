"""Tests for walk-forward optimization."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardResult,
    aggregate_oos_metrics,
)


def _simple_sma_strategy(close: np.ndarray, fast: int = 10, slow: int = 30) -> np.ndarray:
    """Minimal SMA crossover for testing (no external dependencies)."""
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(slow, n):
        fast_avg = np.mean(close[i - fast : i])
        slow_avg = np.mean(close[i - slow : i])
        positions[i] = 1.0 if fast_avg > slow_avg else -1.0
    return positions


# =============================================================================
# WalkForwardResult Tests
# =============================================================================


class TestWalkForwardResult:
    """Test WalkForwardResult dataclass."""

    def test_default_construction(self):
        """Default construction should produce empty dicts and zero fields."""
        r = WalkForwardResult()
        assert r.is_metrics == {}
        assert r.oos_metrics == {}
        assert r.is_window == 0
        assert r.oos_window == 0
        assert r.optimal_params == {}
        assert r.equity_curve_oos == []

    def test_custom_construction(self):
        """Custom values should be stored correctly."""
        r = WalkForwardResult(
            is_metrics={"sharpe": 1.5},
            oos_metrics={"sharpe": 0.8},
            is_window=756,
            oos_window=252,
            optimal_params={"fast": 20},
            equity_curve_oos=[1.0, 1.01],
        )
        assert r.is_metrics["sharpe"] == 1.5
        assert r.oos_window == 252


# =============================================================================
# WalkForwardOptimizer Tests
# =============================================================================


class TestWalkForwardOptimizer:
    """Test WalkForwardOptimizer."""

    @pytest.fixture
    def long_close(self):
        """Generate 5 years of synthetic price data."""
        rng = np.random.default_rng(42)
        n = 1260  # ~5 years
        returns = rng.normal(0.0005, 0.015, n)
        return 100.0 * np.cumprod(1 + returns)

    def test_produces_results(self, long_close):
        """Should produce at least one WalkForwardResult."""
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=long_close,
            strategy_fn=_simple_sma_strategy,
            param_grid={"fast": [10, 20], "slow": [40, 60]},
            is_window=504,
            oos_window=252,
        )
        assert len(results) >= 1
        assert all(isinstance(r, WalkForwardResult) for r in results)

    def test_result_has_metrics(self, long_close):
        """Each result should have IS and OOS metrics."""
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=long_close,
            strategy_fn=_simple_sma_strategy,
            param_grid={"fast": [10], "slow": [40]},
            is_window=504,
            oos_window=252,
        )
        for r in results:
            assert "sharpe" in r.is_metrics
            assert "sharpe" in r.oos_metrics
            assert "total_return" in r.is_metrics
            assert "total_return" in r.oos_metrics

    def test_optimal_params_selected(self, long_close):
        """optimal_params should be from the param_grid."""
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=long_close,
            strategy_fn=_simple_sma_strategy,
            param_grid={"fast": [10, 20, 30], "slow": [50, 80]},
            is_window=504,
            oos_window=252,
        )
        for r in results:
            assert r.optimal_params["fast"] in [10, 20, 30]
            assert r.optimal_params["slow"] in [50, 80]

    def test_equity_curve_starts_at_one(self, long_close):
        """OOS equity curve should start at 1.0."""
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=long_close,
            strategy_fn=_simple_sma_strategy,
            param_grid={"fast": [10], "slow": [40]},
            is_window=504,
            oos_window=252,
        )
        for r in results:
            assert r.equity_curve_oos[0] == 1.0

    def test_insufficient_data_returns_empty(self):
        """Data shorter than IS + OOS should return empty list."""
        close = np.linspace(100, 110, 50)
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=close,
            strategy_fn=_simple_sma_strategy,
            param_grid={"fast": [10], "slow": [30]},
            is_window=504,
            oos_window=252,
        )
        assert results == []

    def test_empty_param_grid_returns_empty(self, long_close):
        """Empty param grid should return empty results."""
        wfo = WalkForwardOptimizer()
        results = wfo.run(
            close=long_close,
            strategy_fn=_simple_sma_strategy,
            param_grid={},
            is_window=504,
            oos_window=252,
        )
        assert results == []


# =============================================================================
# aggregate_oos_metrics Tests
# =============================================================================


class TestAggregateOOSMetrics:
    """Test OOS metrics aggregation."""

    def test_empty_results(self):
        """Empty results should return zeroed metrics."""
        agg = aggregate_oos_metrics([])
        assert agg["n_windows"] == 0
        assert agg["mean_sharpe"] == 0.0

    def test_single_result(self):
        """Single result should be its own aggregate."""
        r = WalkForwardResult(
            is_metrics={"sharpe": 2.0, "total_return": 0.10},
            oos_metrics={"sharpe": 1.0, "total_return": 0.05},
        )
        agg = aggregate_oos_metrics([r])
        assert agg["n_windows"] == 1
        assert agg["mean_sharpe"] == pytest.approx(1.0)
        assert agg["is_oos_sharpe_gap"] == pytest.approx(1.0)  # IS - OOS

    def test_multiple_results(self):
        """Multiple results should be averaged correctly."""
        results = [
            WalkForwardResult(
                is_metrics={"sharpe": 2.0, "total_return": 0.10},
                oos_metrics={"sharpe": 1.0, "total_return": 0.05},
            ),
            WalkForwardResult(
                is_metrics={"sharpe": 1.5, "total_return": 0.08},
                oos_metrics={"sharpe": 0.5, "total_return": 0.03},
            ),
        ]
        agg = aggregate_oos_metrics(results)
        assert agg["n_windows"] == 2
        assert agg["mean_sharpe"] == pytest.approx(0.75)
        assert agg["is_oos_sharpe_gap"] == pytest.approx(1.0)  # (2+1.5)/2 - (1+0.5)/2
