"""Tests for risk analyzer (synthetic data, no API calls)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.models import RiskMetrics
from quant_tools.portfolio.risk import RiskAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def normal_returns(rng) -> np.ndarray:
    """Standard normal-ish daily returns (252 days)."""
    return rng.normal(0.0003, 0.015, 252)


@pytest.fixture
def all_positive_returns(rng) -> np.ndarray:
    """All positive returns."""
    return np.abs(rng.normal(0.001, 0.005, 252))


@pytest.fixture
def all_negative_returns(rng) -> np.ndarray:
    """All negative returns."""
    return -np.abs(rng.normal(0.001, 0.005, 252))


@pytest.fixture
def few_returns() -> np.ndarray:
    """Too few returns for meaningful VaR."""
    return np.array([0.01, -0.02, 0.005])


@pytest.fixture
def risk() -> RiskAnalyzer:
    return RiskAnalyzer()


# ── VaR Tests ─────────────────────────────────────────────────────────


class TestVaR:
    def test_var_is_negative_for_normal_returns(self, risk, normal_returns):
        var = risk.calculate_var(normal_returns)
        assert var < 0

    def test_var_with_known_distribution(self, risk):
        """With uniform [-1, 1], 95% VaR should be near -0.9."""
        uniform = np.linspace(-1, 1, 1000)
        var = risk.calculate_var(uniform, confidence=0.95)
        assert -1.0 < var < -0.85

    def test_var_higher_confidence_more_extreme(self, risk, normal_returns):
        """99% VaR should be more extreme (more negative) than 95% VaR."""
        var_95 = risk.calculate_var(normal_returns, confidence=0.95)
        var_99 = risk.calculate_var(normal_returns, confidence=0.99)
        assert var_99 < var_95

    def test_var_too_few_returns(self, risk, few_returns):
        assert risk.calculate_var(few_returns) == 0.0


# ── CVaR Tests ────────────────────────────────────────────────────────


class TestCVaR:
    def test_cvar_more_extreme_than_var(self, risk, normal_returns):
        """CVaR (expected shortfall) should be <= VaR."""
        var = risk.calculate_var(normal_returns)
        cvar = risk.calculate_cvar(normal_returns)
        assert cvar <= var

    def test_cvar_negative_for_normal_returns(self, risk, normal_returns):
        cvar = risk.calculate_cvar(normal_returns)
        assert cvar < 0

    def test_cvar_too_few_returns(self, risk, few_returns):
        assert risk.calculate_cvar(few_returns) == 0.0


# ── Kelly Tests ───────────────────────────────────────────────────────


class TestKelly:
    def test_kelly_basic(self, risk):
        # 60% win rate, 2:1 win/loss ratio
        kelly = risk.calculate_kelly(win_rate=0.6, avg_win=0.02, avg_loss=0.01)
        assert 0.0 < kelly <= 0.25

    def test_kelly_caps_at_25_percent(self, risk):
        # Very favorable: 90% win rate, 10:1 ratio
        kelly = risk.calculate_kelly(win_rate=0.9, avg_win=0.10, avg_loss=0.01)
        assert kelly == 0.25

    def test_kelly_zero_for_bad_odds(self, risk):
        # 30% win rate, 1:1 ratio -> negative Kelly -> capped to 0
        kelly = risk.calculate_kelly(win_rate=0.3, avg_win=0.01, avg_loss=0.01)
        assert kelly == 0.0

    def test_kelly_zero_for_invalid_inputs(self, risk):
        assert risk.calculate_kelly(win_rate=0.0, avg_win=0.01, avg_loss=0.01) == 0.0
        assert risk.calculate_kelly(win_rate=1.0, avg_win=0.01, avg_loss=0.01) == 0.0
        assert risk.calculate_kelly(win_rate=0.5, avg_win=0.0, avg_loss=0.01) == 0.0
        assert risk.calculate_kelly(win_rate=0.5, avg_win=0.01, avg_loss=0.0) == 0.0
        assert risk.calculate_kelly(win_rate=0.5, avg_win=-0.01, avg_loss=0.01) == 0.0

    def test_kelly_fifty_fifty_even(self, risk):
        # 50% win, 1:1 ratio -> Kelly = 0
        kelly = risk.calculate_kelly(win_rate=0.5, avg_win=0.01, avg_loss=0.01)
        assert kelly == 0.0


# ── Max Position Tests ────────────────────────────────────────────────


class TestMaxPosition:
    def test_basic_max_position(self, risk):
        # VaR = -5%, max loss = 2% of 100k = 2000, position = 2000/0.05 = 40000
        pos = risk.calculate_max_position(100_000, var_95=-0.05)
        assert abs(pos - 40_000) < 1.0

    def test_max_position_capped_at_portfolio(self, risk):
        # Very small VaR -> position would exceed portfolio
        pos = risk.calculate_max_position(100_000, var_95=-0.0001)
        assert pos <= 100_000

    def test_max_position_positive_var(self, risk):
        # Positive VaR (no loss) -> return portfolio value
        pos = risk.calculate_max_position(100_000, var_95=0.01)
        assert pos == 100_000

    def test_max_position_zero_portfolio(self, risk):
        pos = risk.calculate_max_position(0, var_95=-0.05)
        assert pos == 0


# ── Full Report Tests ─────────────────────────────────────────────────


class TestFullReport:
    def test_returns_risk_metrics(self, risk, normal_returns):
        result = risk.full_risk_report(normal_returns)
        assert isinstance(result, RiskMetrics)

    def test_var_negative_in_report(self, risk, normal_returns):
        result = risk.full_risk_report(normal_returns)
        assert result.var_95 < 0

    def test_cvar_more_extreme_in_report(self, risk, normal_returns):
        result = risk.full_risk_report(normal_returns)
        assert result.cvar_95 <= result.var_95

    def test_kelly_in_range(self, risk, normal_returns):
        result = risk.full_risk_report(normal_returns)
        assert 0.0 <= result.kelly_fraction <= 0.25

    def test_max_position_positive(self, risk, normal_returns):
        result = risk.full_risk_report(normal_returns)
        assert result.max_position_size > 0

    def test_all_positive_returns(self, risk, all_positive_returns):
        result = risk.full_risk_report(all_positive_returns)
        # No losses -> Kelly should be 0 (no losing trades to calculate from)
        assert result.kelly_fraction == 0.0

    def test_all_negative_returns(self, risk, all_negative_returns):
        result = risk.full_risk_report(all_negative_returns)
        assert result.kelly_fraction == 0.0
        assert result.var_95 < 0

    def test_pandas_series_input(self, risk, rng):
        import pandas as pd
        s = pd.Series(rng.normal(0.0003, 0.015, 252))
        result = risk.full_risk_report(s)
        assert isinstance(result, RiskMetrics)
