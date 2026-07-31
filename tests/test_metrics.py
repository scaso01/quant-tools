"""Tests for backtest/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.metrics import (
    calculate_alpha,
    calculate_beta,
    calculate_cagr,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_sortino,
    calculate_total_return,
    calculate_tracking_error,
    calculate_treynor,
    calculate_win_rate,
)


class TestTotalReturn:
    """Test calculate_total_return."""

    def test_known_return(self):
        equity = np.array([100.0, 110.0, 120.0, 115.0])
        assert calculate_total_return(equity) == pytest.approx(0.15, abs=1e-9)

    def test_zero_return(self):
        equity = np.array([100.0, 100.0, 100.0])
        assert calculate_total_return(equity) == pytest.approx(0.0)

    def test_negative_return(self):
        equity = np.array([100.0, 90.0, 80.0])
        assert calculate_total_return(equity) == pytest.approx(-0.20, abs=1e-9)

    def test_empty_array(self):
        assert calculate_total_return(np.array([])) == 0.0

    def test_single_element(self):
        assert calculate_total_return(np.array([100.0])) == 0.0

    def test_starting_at_zero(self):
        equity = np.array([0.0, 100.0])
        assert calculate_total_return(equity) == 0.0

    def test_doubling(self):
        equity = np.array([50.0, 100.0])
        assert calculate_total_return(equity) == pytest.approx(1.0, abs=1e-9)


class TestCAGR:
    """Test calculate_cagr."""

    def test_constant_equity(self):
        # No growth => CAGR = 0
        equity = np.full(253, 100.0)
        assert calculate_cagr(equity) == pytest.approx(0.0, abs=1e-9)

    def test_one_year_doubling(self):
        # Start at 100, end at 200, over 252 periods (1 year)
        equity = np.linspace(100, 200, 253)
        # CAGR = (200/100)^(1/1) - 1 = 1.0 (100%)
        assert calculate_cagr(equity) == pytest.approx(1.0, abs=1e-6)

    def test_two_year_growth(self):
        # Start at 100, end at 121, over 504 periods (2 years)
        # CAGR = (121/100)^(1/2) - 1 = 0.10 (10%)
        equity = np.linspace(100, 121, 505)
        assert calculate_cagr(equity) == pytest.approx(0.10, abs=1e-3)

    def test_empty_array(self):
        assert calculate_cagr(np.array([])) == 0.0

    def test_single_element(self):
        assert calculate_cagr(np.array([100.0])) == 0.0

    def test_negative_start(self):
        equity = np.array([-100.0, 200.0])
        assert calculate_cagr(equity) == 0.0


class TestSharpe:
    """Test calculate_sharpe."""

    def test_zero_std_returns(self):
        # All same return => std=0 => Sharpe=0
        returns = np.full(100, 0.001)
        assert calculate_sharpe(returns) == 0.0

    def test_positive_excess_returns(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        # With mean=0.001 and std=0.01, Sharpe should be ~1.59 annualized
        assert sharpe > 0

    def test_negative_excess_returns(self):
        np.random.seed(42)
        returns = np.random.normal(-0.002, 0.01, 252)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        assert sharpe < 0

    def test_empty_returns(self):
        assert calculate_sharpe(np.array([])) == 0.0

    def test_single_return(self):
        assert calculate_sharpe(np.array([0.01])) == 0.0


class TestSortino:
    """Test calculate_sortino."""

    def test_no_downside(self):
        # All positive returns => no downside => 0 (no negative excess)
        returns = np.full(100, 0.01)
        # With rf ~0.04/252 ~ 0.000159, 0.01 is always positive excess
        assert calculate_sortino(returns, risk_free_rate=0.0) == 0.0

    def test_mixed_returns(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.015, 252)
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        # Sortino should be positive for positive mean
        assert sortino > 0

    def test_empty_returns(self):
        assert calculate_sortino(np.array([])) == 0.0

    def test_single_return(self):
        assert calculate_sortino(np.array([0.01])) == 0.0

    def test_sortino_gte_sharpe_for_positive_skew(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        # Sortino divides by downside deviation only, so it should come out at
        # or above Sharpe. That does not hold for every sample, but the seed is
        # fixed, so this one is decided: 1.6833 against 1.5795. The test used to
        # assert only that sortino was non-zero, which its name did not promise
        # and which would have passed with the two measures the wrong way round.
        assert sortino >= sharpe


class TestMaxDrawdown:
    """Test calculate_max_drawdown."""

    def test_no_drawdown(self):
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        dd, dur = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.0)
        assert dur == 0

    def test_known_drawdown(self):
        # Peak at 200, drops to 150 = 25% drawdown
        equity = np.array([100.0, 200.0, 150.0, 180.0])
        dd, dur = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.25, abs=1e-9)

    def test_full_recovery(self):
        equity = np.array([100.0, 200.0, 150.0, 100.0, 200.0])
        dd, dur = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.50, abs=1e-9)
        assert dur == 2  # In drawdown for 2 periods before recovery

    def test_unrecovered_drawdown(self):
        equity = np.array([100.0, 200.0, 100.0, 120.0])
        dd, dur = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.50, abs=1e-9)
        # Duration = 2 periods from peak (never recovered)
        assert dur >= 2

    def test_empty_array(self):
        dd, dur = calculate_max_drawdown(np.array([]))
        assert dd == 0.0
        assert dur == 0

    def test_single_element(self):
        dd, dur = calculate_max_drawdown(np.array([100.0]))
        assert dd == 0.0
        assert dur == 0

    def test_monotonic_decline(self):
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd, dur = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.30, abs=1e-9)
        assert dur == 3


class TestWinRate:
    """Test calculate_win_rate."""

    def test_all_winners(self):
        trades = np.array([0.05, 0.03, 0.10])
        assert calculate_win_rate(trades) == pytest.approx(1.0)

    def test_all_losers(self):
        trades = np.array([-0.05, -0.03, -0.10])
        assert calculate_win_rate(trades) == pytest.approx(0.0)

    def test_mixed(self):
        trades = np.array([0.05, -0.03, 0.10, -0.02])
        assert calculate_win_rate(trades) == pytest.approx(0.5)

    def test_empty(self):
        assert calculate_win_rate(np.array([])) == 0.0

    def test_zero_returns_not_winners(self):
        trades = np.array([0.0, 0.0, 0.05])
        assert calculate_win_rate(trades) == pytest.approx(1.0 / 3.0, abs=1e-9)


class TestProfitFactor:
    """Test calculate_profit_factor."""

    def test_balanced(self):
        trades = np.array([0.10, -0.05, 0.10, -0.05])
        # gross profit = 0.20, gross loss = 0.10
        assert calculate_profit_factor(trades) == pytest.approx(2.0, abs=1e-9)

    def test_all_profits(self):
        trades = np.array([0.05, 0.10])
        assert calculate_profit_factor(trades) == float("inf")

    def test_all_losses(self):
        trades = np.array([-0.05, -0.10])
        assert calculate_profit_factor(trades) == pytest.approx(0.0)

    def test_empty(self):
        assert calculate_profit_factor(np.array([])) == 0.0

    def test_all_zeros(self):
        trades = np.array([0.0, 0.0])
        assert calculate_profit_factor(trades) == 0.0


# ── Benchmark-Relative Metric Tests ────────────────────────────────


class TestBeta:
    """Test calculate_beta."""

    def test_identity_returns(self):
        """Same returns => beta = 1.0."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        assert calculate_beta(returns, returns) == pytest.approx(1.0, abs=1e-9)

    def test_doubled_returns(self):
        """Portfolio = 2x benchmark => beta = 2.0."""
        np.random.seed(42)
        bench = np.random.normal(0.001, 0.01, 252)
        port = bench * 2.0
        assert calculate_beta(port, bench) == pytest.approx(2.0, abs=1e-9)

    def test_uncorrelated(self):
        """Independent returns => beta ~ 0."""
        np.random.seed(42)
        port = np.random.normal(0.001, 0.01, 1000)
        np.random.seed(99)
        bench = np.random.normal(0.001, 0.01, 1000)
        beta = calculate_beta(port, bench)
        assert abs(beta) < 0.15

    def test_negative_beta(self):
        """Negated returns => beta = -1.0."""
        np.random.seed(42)
        bench = np.random.normal(0.001, 0.01, 252)
        port = -bench
        assert calculate_beta(port, bench) == pytest.approx(-1.0, abs=1e-9)

    def test_empty_returns(self):
        assert calculate_beta(np.array([]), np.array([])) == 0.0

    def test_single_return(self):
        assert calculate_beta(np.array([0.01]), np.array([0.01])) == 0.0

    def test_zero_variance_benchmark(self):
        """Constant benchmark => beta = 0 (undefined)."""
        port = np.random.normal(0.001, 0.01, 100)
        bench = np.full(100, 0.001)
        assert calculate_beta(port, bench) == 0.0


class TestAlpha:
    """Test calculate_alpha."""

    def test_identity_returns(self):
        """Same returns as benchmark => alpha ~ 0."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        alpha = calculate_alpha(returns, returns, risk_free_rate=0.04)
        assert abs(alpha) < 0.01

    def test_outperformance(self):
        """Portfolio consistently above benchmark => positive alpha."""
        np.random.seed(42)
        bench = np.random.normal(0.0004, 0.01, 252)
        port = bench + 0.001  # 0.1% daily outperformance
        alpha = calculate_alpha(port, bench, risk_free_rate=0.04)
        assert alpha > 0.1  # Should be ~25% annualized

    def test_underperformance(self):
        """Portfolio consistently below benchmark => negative alpha."""
        np.random.seed(42)
        bench = np.random.normal(0.0004, 0.01, 252)
        port = bench - 0.001
        alpha = calculate_alpha(port, bench, risk_free_rate=0.04)
        assert alpha < -0.1

    def test_empty_returns(self):
        assert calculate_alpha(np.array([]), np.array([])) == 0.0


class TestTrackingError:
    """Test calculate_tracking_error."""

    def test_identity_returns(self):
        """Same returns => TE = 0."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        te = calculate_tracking_error(returns, returns)
        assert te == pytest.approx(0.0, abs=1e-12)

    def test_positive_te(self):
        """Different returns => positive TE."""
        np.random.seed(42)
        port = np.random.normal(0.001, 0.015, 252)
        np.random.seed(99)
        bench = np.random.normal(0.001, 0.01, 252)
        te = calculate_tracking_error(port, bench)
        assert te > 0

    def test_scaled_difference(self):
        """Constant offset => TE = 0 (tracking error is on diff volatility)."""
        np.random.seed(42)
        bench = np.random.normal(0.001, 0.01, 252)
        port = bench + 0.001  # constant shift, same volatility pattern
        te = calculate_tracking_error(port, bench)
        assert te == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns(self):
        assert calculate_tracking_error(np.array([]), np.array([])) == 0.0


class TestInformationRatio:
    """Test calculate_information_ratio."""

    def test_identity_returns(self):
        """Same returns => IR = 0."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        ir = calculate_information_ratio(returns, returns)
        assert ir == pytest.approx(0.0, abs=1e-9)

    def test_outperformance(self):
        """Consistent outperformance => positive IR."""
        np.random.seed(42)
        bench = np.random.normal(0.0004, 0.01, 252)
        port = bench + np.random.normal(0.001, 0.002, 252)
        ir = calculate_information_ratio(port, bench)
        assert ir > 0

    def test_underperformance(self):
        """Consistent underperformance => negative IR."""
        np.random.seed(42)
        bench = np.random.normal(0.0004, 0.01, 252)
        port = bench + np.random.normal(-0.001, 0.002, 252)
        ir = calculate_information_ratio(port, bench)
        assert ir < 0

    def test_empty_returns(self):
        assert calculate_information_ratio(np.array([]), np.array([])) == 0.0


class TestTreynor:
    """Test calculate_treynor."""

    def test_positive_treynor(self):
        """Portfolio with positive return above rf and beta=1 => positive Treynor."""
        np.random.seed(42)
        # ~20% annualized return, beta ~1
        bench = np.random.normal(0.001, 0.01, 252)
        port = bench + np.random.normal(0.0003, 0.002, 252)
        treynor = calculate_treynor(port, bench, risk_free_rate=0.04)
        assert treynor > 0

    def test_zero_beta(self):
        """Zero beta => Treynor = 0 (undefined)."""
        port = np.full(100, 0.001)
        bench = np.full(100, 0.001)
        treynor = calculate_treynor(port, bench)
        assert treynor == 0.0

    def test_doubled_returns(self):
        """Higher beta should reduce Treynor ratio for same excess return."""
        np.random.seed(42)
        bench = np.random.normal(0.001, 0.01, 252)
        port1 = bench * 1.0 + 0.001
        port2 = bench * 2.0 + 0.001  # same alpha, double beta
        t1 = calculate_treynor(port1, bench, risk_free_rate=0.0)
        t2 = calculate_treynor(port2, bench, risk_free_rate=0.0)
        # port2 has 2x beta but not 2x excess return
        assert t1 != t2

    def test_empty_returns(self):
        assert calculate_treynor(np.array([]), np.array([])) == 0.0
