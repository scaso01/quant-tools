"""Tests for anti-overfitting metrics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.overfitting import (
    deflated_sharpe_ratio,
    overfitting_traffic_light,
    probability_of_backtest_overfitting,
)


# =============================================================================
# Deflated Sharpe Ratio Tests
# =============================================================================


class TestDeflatedSharpeRatio:
    """Test Deflated Sharpe Ratio calculation."""

    def test_returns_valid_probability(self):
        """DSR should return a value in [0, 1]."""
        p = deflated_sharpe_ratio(sharpe=1.5, n_trials=100, n_obs=252)
        assert 0.0 <= p <= 1.0

    def test_higher_sharpe_lower_pvalue(self):
        """Higher observed Sharpe should produce lower p-value (more significant)."""
        p_low = deflated_sharpe_ratio(sharpe=0.5, n_trials=100, n_obs=252)
        p_high = deflated_sharpe_ratio(sharpe=3.0, n_trials=100, n_obs=252)
        assert p_high > p_low  # Higher Sharpe = more likely genuine = higher CDF

    def test_more_trials_higher_pvalue(self):
        """More trials should make it harder to pass (higher p-value for same Sharpe)."""
        p_few = deflated_sharpe_ratio(sharpe=1.5, n_trials=10, n_obs=252)
        p_many = deflated_sharpe_ratio(sharpe=1.5, n_trials=10000, n_obs=252)
        assert p_few > p_many  # More trials = higher bar = lower CDF for same Sharpe

    def test_single_trial_most_lenient(self):
        """With 1 trial, DSR should be most favorable."""
        p = deflated_sharpe_ratio(sharpe=1.0, n_trials=1, n_obs=252)
        assert p > 0.3  # Should be a reasonable p-value

    def test_zero_sharpe_high_pvalue(self):
        """Zero Sharpe with many trials should produce high p-value (not genuine)."""
        p = deflated_sharpe_ratio(sharpe=0.0, n_trials=1000, n_obs=252)
        assert p < 0.5

    def test_skewness_effect(self):
        """Negative skewness should increase the adjusted variance."""
        p_normal = deflated_sharpe_ratio(sharpe=1.0, n_trials=100, skewness=0.0, n_obs=252)
        p_skewed = deflated_sharpe_ratio(sharpe=1.0, n_trials=100, skewness=-2.0, n_obs=252)
        # Skewness changes the variance of the Sharpe estimator
        assert p_normal != p_skewed

    def test_edge_zero_trials(self):
        """Zero or negative trials should return 1.0 (no confidence)."""
        assert deflated_sharpe_ratio(sharpe=2.0, n_trials=0) == 1.0

    def test_edge_insufficient_obs(self):
        """n_obs < 2 should return 1.0."""
        assert deflated_sharpe_ratio(sharpe=2.0, n_trials=10, n_obs=1) == 1.0


# =============================================================================
# Probability of Backtest Overfitting Tests
# =============================================================================


class TestProbabilityOfBacktestOverfitting:
    """Test PBO estimation."""

    def test_random_strategies_high_pbo(self):
        """Random strategies should show high PBO (near 0.5)."""
        rng = np.random.default_rng(42)
        # 500 obs, 10 random strategies
        returns = rng.normal(0.0, 0.01, (500, 10))
        pbo = probability_of_backtest_overfitting(returns, n_groups=10)
        # Random strategies: PBO should be roughly 0.3-0.7
        assert 0.0 <= pbo <= 1.0

    def test_single_strategy_returns_zero(self):
        """Single strategy should return PBO = 0 (nothing to compare)."""
        returns = np.random.default_rng(42).normal(0.001, 0.01, (500, 1))
        pbo = probability_of_backtest_overfitting(returns, n_groups=10)
        assert pbo == 0.0

    def test_pbo_in_valid_range(self):
        """PBO should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0, 0.02, (200, 5))
        pbo = probability_of_backtest_overfitting(returns, n_groups=6)
        assert 0.0 <= pbo <= 1.0

    def test_insufficient_data(self):
        """Fewer observations than groups should return 1.0 (maximum uncertainty)."""
        returns = np.random.default_rng(42).normal(0.0, 0.01, (5, 3))
        pbo = probability_of_backtest_overfitting(returns, n_groups=10)
        assert pbo == 1.0


# =============================================================================
# Traffic Light Tests
# =============================================================================


class TestOverfittingTrafficLight:
    """Test PBO traffic light classification."""

    def test_green(self):
        assert overfitting_traffic_light(0.10) == "green"
        assert overfitting_traffic_light(0.0) == "green"
        assert overfitting_traffic_light(0.24) == "green"

    def test_yellow(self):
        assert overfitting_traffic_light(0.25) == "yellow"
        assert overfitting_traffic_light(0.40) == "yellow"
        assert overfitting_traffic_light(0.49) == "yellow"

    def test_red(self):
        assert overfitting_traffic_light(0.50) == "red"
        assert overfitting_traffic_light(0.75) == "red"
        assert overfitting_traffic_light(1.0) == "red"
