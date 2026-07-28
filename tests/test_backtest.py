"""Tests for backtest engine, strategies, and full pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.engine import run_backtest
from quant_tools.backtest.strategies import (
    bollinger_strategy,
    factor_strategy,
    momentum_strategy,
    rsi_strategy,
    sma_crossover,
)
from quant_tools.models import BacktestResult, StrategyType


# =============================================================================
# Engine Tests
# =============================================================================


class TestRunBacktest:
    """Test the backtest engine."""

    def test_all_long_matches_buy_and_hold(self, sample_close_array):
        """All-long position should approximate buy-and-hold return."""
        close = sample_close_array
        positions = np.ones(len(close), dtype=np.float64)

        result = run_backtest(close, positions, initial_capital=100000, commission=0.0)

        expected_return = (close[-1] / close[0]) - 1.0
        assert result.total_return == pytest.approx(expected_return, rel=0.01)
        assert isinstance(result, BacktestResult)

    def test_all_cash_returns_initial(self, sample_close_array):
        """All-cash (position=0) should keep capital ~unchanged."""
        close = sample_close_array
        positions = np.zeros(len(close), dtype=np.float64)

        result = run_backtest(close, positions, initial_capital=100000, commission=0.0)

        assert result.final_value == pytest.approx(100000.0, abs=1.0)
        assert result.total_return == pytest.approx(0.0, abs=1e-9)

    def test_commission_reduces_returns(self, sample_close_array):
        """Commission should reduce final value compared to no commission."""
        close = sample_close_array
        positions = np.ones(len(close), dtype=np.float64)
        # Add some position changes to trigger commissions
        positions[::50] = 0  # Go flat every 50 bars

        result_no_comm = run_backtest(close, positions, commission=0.0)
        result_with_comm = run_backtest(close, positions, commission=0.005)

        assert result_with_comm.final_value < result_no_comm.final_value

    def test_result_has_correct_fields(self, sample_close_array):
        """Result should have all expected fields populated."""
        close = sample_close_array
        positions = np.ones(len(close), dtype=np.float64)

        result = run_backtest(
            close, positions,
            strategy=StrategyType.SMA_CROSSOVER,
            ticker="TEST",
        )

        assert result.strategy == StrategyType.SMA_CROSSOVER
        assert result.ticker == "TEST"
        assert result.initial_capital == 100000.0
        assert len(result.equity_curve) == len(close)
        assert result.equity_curve[0] == 100000.0

    def test_short_input(self):
        """Single-element input should return empty result."""
        close = np.array([100.0])
        positions = np.array([1.0])
        result = run_backtest(close, positions)
        assert result.total_return == 0.0
        assert result.final_value == 100000.0

    def test_equity_curve_monotonic_for_all_long_in_uptrend(self):
        """In a strictly rising market, all-long should be monotonically increasing (no commission)."""
        close = np.linspace(100, 200, 100)
        positions = np.ones(100, dtype=np.float64)
        result = run_backtest(close, positions, commission=0.0)
        equity = np.array(result.equity_curve)
        diffs = np.diff(equity)
        assert np.all(diffs >= -1e-10)  # Allow tiny float rounding

    def test_max_drawdown_populated(self, sample_close_array):
        """Max drawdown should be >= 0."""
        close = sample_close_array
        positions = np.ones(len(close), dtype=np.float64)
        result = run_backtest(close, positions)
        assert result.max_drawdown >= 0.0

    def test_sharpe_is_finite(self, sample_close_array):
        """Sharpe ratio should be a finite number."""
        close = sample_close_array
        positions = np.ones(len(close), dtype=np.float64)
        result = run_backtest(close, positions)
        assert np.isfinite(result.sharpe_ratio)


# =============================================================================
# Strategy Tests
# =============================================================================


class TestSMACrossover:
    """Test SMA crossover strategy signal generation."""

    def test_valid_signals(self, sample_close_array):
        """Signals should be in {-1, 0, 1}."""
        positions = sma_crossover(sample_close_array, fast=10, slow=30)
        unique = set(np.unique(positions))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_warmup_period_is_zero(self, sample_close_array):
        """During warmup (before slow SMA is valid), positions should be 0."""
        positions = sma_crossover(sample_close_array, fast=10, slow=30)
        # First 29 elements should be 0 (slow SMA needs 30 periods)
        assert all(positions[:29] == 0.0)

    def test_signal_after_warmup(self, sample_close_array):
        """After warmup, positions should be non-zero."""
        positions = sma_crossover(sample_close_array, fast=10, slow=30)
        assert any(positions[30:] != 0.0)

    def test_output_length(self, sample_close_array):
        """Output should be same length as input."""
        positions = sma_crossover(sample_close_array)
        assert len(positions) == len(sample_close_array)


class TestBollingerStrategy:
    """Test Bollinger Bands strategy signal generation."""

    def test_valid_signals(self, sample_close_array):
        positions = bollinger_strategy(sample_close_array)
        unique = set(np.unique(positions))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_output_length(self, sample_close_array):
        positions = bollinger_strategy(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_warmup_zeros(self, sample_close_array):
        positions = bollinger_strategy(sample_close_array, period=20)
        # First 19 should be 0
        assert all(positions[:19] == 0.0)


class TestRSIStrategy:
    """Test RSI strategy signal generation."""

    def test_valid_signals(self, sample_close_array):
        positions = rsi_strategy(sample_close_array)
        unique = set(np.unique(positions))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_output_length(self, sample_close_array):
        positions = rsi_strategy(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_all_up_goes_short(self):
        """Monotonically increasing prices should trigger overbought (short)."""
        close = np.arange(50.0, 200.0, 1.0)
        positions = rsi_strategy(close, period=14, oversold=30, overbought=70)
        # Near the end, RSI should be very high => short
        assert positions[-1] == -1.0


class TestMomentumStrategy:
    """Test momentum strategy signal generation."""

    def test_valid_signals(self, sample_close_array):
        positions = momentum_strategy(sample_close_array, lookback=30)
        unique = set(np.unique(positions))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_warmup_zeros(self, sample_close_array):
        positions = momentum_strategy(sample_close_array, lookback=30)
        assert all(positions[:30] == 0.0)

    def test_uptrend_goes_long(self):
        close = np.linspace(100, 200, 100)
        positions = momentum_strategy(close, lookback=20, threshold=0.0)
        # After warmup, should be long
        assert all(positions[20:] == 1.0)


class TestFactorStrategy:
    """Test factor-based strategy signal generation."""

    def test_valid_signals(self):
        close = np.linspace(100, 200, 50)
        scores = np.random.default_rng(42).random(50)
        positions = factor_strategy(close, scores, threshold=0.5)
        unique = set(np.unique(positions))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_high_scores_go_long(self):
        close = np.full(10, 100.0)
        scores = np.full(10, 0.9)
        positions = factor_strategy(close, scores, threshold=0.5)
        assert all(positions == 1.0)

    def test_low_scores_go_short(self):
        close = np.full(10, 100.0)
        scores = np.full(10, 0.1)
        positions = factor_strategy(close, scores, threshold=0.5)
        assert all(positions == -1.0)


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Test strategy -> engine -> result pipeline."""

    def test_sma_pipeline(self, sample_close_array):
        """SMA crossover -> engine -> valid result."""
        close = sample_close_array
        positions = sma_crossover(close, fast=10, slow=30)
        result = run_backtest(close, positions, initial_capital=50000)

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 50000.0
        assert result.final_value > 0
        assert len(result.equity_curve) == len(close)
        assert np.isfinite(result.sharpe_ratio)
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_bollinger_pipeline(self, sample_close_array):
        """Bollinger -> engine -> valid result."""
        close = sample_close_array
        positions = bollinger_strategy(close)
        result = run_backtest(
            close, positions,
            strategy=StrategyType.BOLLINGER,
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy == StrategyType.BOLLINGER
        assert result.final_value > 0

    def test_rsi_pipeline(self, sample_close_array):
        """RSI -> engine -> valid result."""
        close = sample_close_array
        positions = rsi_strategy(close)
        result = run_backtest(
            close, positions,
            strategy=StrategyType.RSI,
        )
        assert isinstance(result, BacktestResult)
        assert result.final_value > 0

    def test_momentum_pipeline(self, sample_close_array):
        """Momentum -> engine -> valid result."""
        close = sample_close_array
        positions = momentum_strategy(close, lookback=30)
        result = run_backtest(
            close, positions,
            strategy=StrategyType.MOMENTUM,
        )
        assert isinstance(result, BacktestResult)
        assert result.final_value > 0

    def test_factor_pipeline(self, sample_close_array):
        """Factor -> engine -> valid result."""
        close = sample_close_array
        scores = np.random.default_rng(42).random(len(close))
        positions = factor_strategy(close, scores, threshold=0.5)
        result = run_backtest(
            close, positions,
            strategy=StrategyType.FACTOR,
        )
        assert isinstance(result, BacktestResult)
        assert result.final_value > 0
