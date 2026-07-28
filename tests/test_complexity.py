"""Tests for model complexity assessment."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.complexity import (
    bayesian_information_criterion,
    complexity_budget,
    count_free_params,
)


# =============================================================================
# count_free_params Tests
# =============================================================================


class TestCountFreeParams:
    """Test parameter counting."""

    def test_known_strategy_sma(self):
        """SMA crossover should have 2 params (fast, slow)."""
        n = count_free_params("sma_crossover", {"fast": 20, "slow": 50})
        assert n == 2

    def test_known_strategy_rsi(self):
        """RSI strategy should have 3 params."""
        n = count_free_params("rsi_strategy", {"period": 14, "oversold": 30, "overbought": 70})
        assert n == 3

    def test_known_strategy_mean_reversion(self):
        """Mean reversion should have 3 params."""
        n = count_free_params("mean_reversion", {"period": 20, "z_entry": 2.0, "z_exit": 0.5})
        assert n == 3

    def test_unknown_strategy_counts_params(self):
        """Unknown strategy should count all params in the dict."""
        n = count_free_params("my_custom_strategy", {"a": 1, "b": 2, "c": 3, "d": 4})
        assert n == 4

    def test_empty_params_for_known(self):
        """Known strategy param count is independent of the params dict."""
        n = count_free_params("sma_crossover", {})
        assert n == 2  # Known count from registry

    def test_zero_params_for_carry_trade(self):
        """Carry trade has no tunable params in the registry."""
        n = count_free_params("carry_trade", {})
        assert n == 0


# =============================================================================
# complexity_budget Tests
# =============================================================================


class TestComplexityBudget:
    """Test complexity budget assessment."""

    def test_sufficient_data_low_risk(self):
        """Plenty of data should yield low risk."""
        result = complexity_budget(n_params=2, n_observations=1000, min_obs_per_param=100)
        assert result["risk_level"] == "low"
        assert result["ratio"] == 500.0

    def test_borderline_medium_risk(self):
        """Borderline data should yield medium risk."""
        # ratio = 75, which is >= 50 (min/2) but < 100
        result = complexity_budget(n_params=2, n_observations=150, min_obs_per_param=100)
        assert result["risk_level"] == "medium"

    def test_insufficient_data_high_risk(self):
        """Too little data should yield high risk."""
        # ratio = 10, which is < 50
        result = complexity_budget(n_params=5, n_observations=50, min_obs_per_param=100)
        assert result["risk_level"] == "high"

    def test_zero_params(self):
        """Zero params should be low risk with infinite ratio."""
        result = complexity_budget(n_params=0, n_observations=100)
        assert result["risk_level"] == "low"
        assert result["ratio"] == float("inf")

    def test_result_keys(self):
        """Result should have all expected keys."""
        result = complexity_budget(n_params=3, n_observations=500)
        expected_keys = {"n_params", "n_obs", "ratio", "warning", "risk_level"}
        assert set(result.keys()) == expected_keys


# =============================================================================
# BIC Tests
# =============================================================================


class TestBayesianInformationCriterion:
    """Test BIC calculation."""

    def test_known_value(self):
        """BIC should match manual calculation."""
        # BIC = -2 * LL + k * ln(n) = -2 * (-100) + 3 * ln(100)
        bic = bayesian_information_criterion(log_likelihood=-100.0, n_params=3, n_obs=100)
        expected = 200.0 + 3 * np.log(100)
        assert bic == pytest.approx(expected, rel=1e-10)

    def test_more_params_higher_bic(self):
        """More parameters should increase BIC (worse score)."""
        bic_few = bayesian_information_criterion(log_likelihood=-100.0, n_params=2, n_obs=100)
        bic_many = bayesian_information_criterion(log_likelihood=-100.0, n_params=10, n_obs=100)
        assert bic_many > bic_few

    def test_invalid_n_obs_raises(self):
        """n_obs < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_obs must be >= 1"):
            bayesian_information_criterion(log_likelihood=-50.0, n_params=2, n_obs=0)
