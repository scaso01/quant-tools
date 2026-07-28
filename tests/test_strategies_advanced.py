"""Tests for advanced strategies."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.strategies_advanced import (
    breakout,
    carry_trade,
    factor_rotation,
    mean_reversion,
    pairs_trading,
    risk_parity_dynamic,
    trend_following,
    volatility_targeting,
)


# =============================================================================
# Helpers
# =============================================================================


def _valid_positions(positions: np.ndarray, allow_fractional: bool = False) -> bool:
    """Check that positions contain only valid values."""
    if allow_fractional:
        return np.all((positions >= -1.0) & (positions <= 1.0))
    unique = set(np.unique(positions))
    return unique.issubset({-1.0, 0.0, 1.0})


# =============================================================================
# Mean Reversion Tests
# =============================================================================


class TestMeanReversion:
    """Test mean reversion strategy."""

    def test_valid_signals(self, sample_close_array):
        positions = mean_reversion(sample_close_array, period=20)
        assert _valid_positions(positions)

    def test_output_length(self, sample_close_array):
        positions = mean_reversion(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_warmup_zeros(self, sample_close_array):
        """First (period-1) positions should be 0."""
        period = 20
        positions = mean_reversion(sample_close_array, period=period)
        assert all(positions[:period - 1] == 0.0)

    def test_extreme_deviation_triggers_entry(self):
        """Price far from mean should trigger entry."""
        # Create price series that drops sharply
        close = np.concatenate([
            np.full(30, 100.0),
            np.linspace(100, 80, 20),  # Sharp drop
        ])
        positions = mean_reversion(close, period=20, z_entry=1.0, z_exit=0.3)
        # After the drop, should be long (mean reversion = buy low)
        assert positions[-1] == 1.0


# =============================================================================
# Pairs Trading Tests
# =============================================================================


class TestPairsTrading:
    """Test pairs trading strategy."""

    def test_valid_signals(self, sample_close_array):
        rng = np.random.default_rng(42)
        close_b = sample_close_array * (1 + rng.normal(0, 0.01, len(sample_close_array)))
        positions = pairs_trading(sample_close_array, close_b, lookback=30)
        assert _valid_positions(positions)

    def test_output_length(self, sample_close_array):
        close_b = sample_close_array * 1.1
        positions = pairs_trading(sample_close_array, close_b, lookback=30)
        assert len(positions) == len(sample_close_array)

    def test_warmup_zeros(self, sample_close_array):
        close_b = sample_close_array * 1.05
        lookback = 60
        positions = pairs_trading(sample_close_array, close_b, lookback=lookback)
        assert all(positions[:lookback - 1] == 0.0)

    def test_divergence_triggers_trade(self):
        """When spreads diverge, should enter position."""
        n = 200
        close_a = np.full(n, 100.0, dtype=np.float64)
        close_b = np.full(n, 100.0, dtype=np.float64)
        # Make B rise sharply relative to A
        close_b[150:] = 120.0
        positions = pairs_trading(close_a, close_b, lookback=60, z_entry=1.5)
        # Spread = log(A) - log(B) becomes very negative -> long A
        assert any(positions[160:] == 1.0)


# =============================================================================
# Trend Following Tests
# =============================================================================


class TestTrendFollowing:
    """Test trend following strategy."""

    def test_valid_signals(self, sample_close_array):
        positions = trend_following(sample_close_array, fast=10, slow=50)
        assert _valid_positions(positions)

    def test_output_length(self, sample_close_array):
        positions = trend_following(sample_close_array, fast=10, slow=50)
        assert len(positions) == len(sample_close_array)

    def test_uptrend_goes_long(self):
        """Steady uptrend should produce long positions."""
        close = np.linspace(100, 200, 200)
        positions = trend_following(close, fast=10, slow=50)
        # After warmup, should eventually be long
        active = positions[60:]
        long_count = np.sum(active == 1.0)
        assert long_count > len(active) * 0.5


# =============================================================================
# Breakout Tests
# =============================================================================


class TestBreakout:
    """Test Donchian channel breakout strategy."""

    def test_valid_signals(self, sample_close_array):
        positions = breakout(sample_close_array, period=20)
        assert _valid_positions(positions)

    def test_output_length(self, sample_close_array):
        positions = breakout(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_warmup_zeros(self, sample_close_array):
        period = 20
        positions = breakout(sample_close_array, period=period)
        assert all(positions[:period] == 0.0)

    def test_breakout_above_goes_long(self):
        """Breaking above channel high should go long."""
        close = np.concatenate([
            np.full(30, 100.0),
            np.array([110.0]),  # Breakout above
        ])
        positions = breakout(close, period=20)
        assert positions[-1] == 1.0

    def test_volume_confirmation(self):
        """Volume below average should prevent breakout entry."""
        close = np.concatenate([
            np.full(30, 100.0),
            np.array([110.0]),
        ])
        volume = np.concatenate([
            np.full(30, 1_000_000.0),
            np.array([100.0]),  # Very low volume
        ])
        positions = breakout(close, volume=volume, period=20)
        # Low volume should prevent the breakout
        assert positions[-1] == 0.0


# =============================================================================
# Factor Rotation Tests
# =============================================================================


class TestFactorRotation:
    """Test factor rotation strategy."""

    def test_with_factor_scores(self, sample_close_array):
        scores = np.random.default_rng(42).normal(0, 1, len(sample_close_array))
        positions = factor_rotation(sample_close_array, factor_scores=scores)
        assert _valid_positions(positions)

    def test_with_regime(self, sample_close_array):
        regime = np.ones(len(sample_close_array))
        regime[::2] = -1
        positions = factor_rotation(sample_close_array, regime=regime)
        assert _valid_positions(positions)

    def test_default_momentum(self, sample_close_array):
        """Without scores or regime, should use momentum."""
        positions = factor_rotation(sample_close_array)
        # First 60 should be warmup
        assert all(positions[:60] == 0.0)
        assert any(positions[60:] != 0.0)


# =============================================================================
# Risk Parity Dynamic Tests
# =============================================================================


class TestRiskParityDynamic:
    """Test dynamic risk parity strategy."""

    def test_positions_in_range(self, sample_close_array):
        """Positions should be in [0, 1]."""
        positions = risk_parity_dynamic(sample_close_array, target_vol=0.15)
        assert _valid_positions(positions, allow_fractional=True)
        assert np.all(positions >= 0.0)
        assert np.all(positions <= 1.0)

    def test_output_length(self, sample_close_array):
        positions = risk_parity_dynamic(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_short_input_all_zeros(self):
        """Very short input should return all zeros."""
        close = np.array([100.0, 101.0, 102.0])
        positions = risk_parity_dynamic(close)
        assert np.all(positions == 0.0)


# =============================================================================
# Carry Trade Tests
# =============================================================================


class TestCarryTrade:
    """Test carry trade strategy."""

    def test_positive_rate_goes_long(self):
        close = np.ones(50)
        rate_diff = np.full(50, 0.02)
        positions = carry_trade(close, rate_diff=rate_diff)
        assert np.all(positions == 1.0)

    def test_negative_rate_goes_short(self):
        close = np.ones(50)
        rate_diff = np.full(50, -0.02)
        positions = carry_trade(close, rate_diff=rate_diff)
        assert np.all(positions == -1.0)

    def test_zero_rate_stays_neutral(self):
        close = np.ones(50)
        rate_diff = np.zeros(50)
        positions = carry_trade(close, rate_diff=rate_diff)
        assert np.all(positions == 0.0)

    def test_default_momentum_proxy(self, sample_close_array):
        """Without rate_diff, should use momentum as proxy."""
        positions = carry_trade(sample_close_array)
        assert _valid_positions(positions)
        assert all(positions[:20] == 0.0)  # Warmup


# =============================================================================
# Volatility Targeting Tests
# =============================================================================


class TestVolatilityTargeting:
    """Test volatility targeting strategy."""

    def test_positions_in_range(self, sample_close_array):
        """Positions should be in [0, 1]."""
        positions = volatility_targeting(sample_close_array, target_vol=0.15, lookback=20)
        assert _valid_positions(positions, allow_fractional=True)
        assert np.all(positions >= 0.0)
        assert np.all(positions <= 1.0)

    def test_output_length(self, sample_close_array):
        positions = volatility_targeting(sample_close_array)
        assert len(positions) == len(sample_close_array)

    def test_high_vol_reduces_position(self):
        """High volatility should produce smaller positions."""
        rng = np.random.default_rng(42)
        # Low vol series
        low_vol = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.005, 200))
        # High vol series
        high_vol = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.05, 200))

        pos_low = volatility_targeting(low_vol, target_vol=0.15, lookback=20)
        pos_high = volatility_targeting(high_vol, target_vol=0.15, lookback=20)

        # After warmup, high vol positions should be smaller on average
        active_low = pos_low[25:]
        active_high = pos_high[25:]
        assert np.mean(active_high[active_high > 0]) < np.mean(active_low[active_low > 0])

    def test_short_input_all_zeros(self):
        """Very short input should return all zeros."""
        close = np.array([100.0, 101.0])
        positions = volatility_targeting(close, lookback=20)
        assert np.all(positions == 0.0)
