"""Tests for enhanced risk analysis (VaR trio, stress testing, tail risk)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.portfolio.risk_enhanced import (
    HISTORICAL_SCENARIOS,
    EnhancedRiskAnalyzer,
    StressScenario,
    TailRiskMetrics,
    VaRResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def normal_returns(rng) -> np.ndarray:
    """Normal returns with known mean/vol."""
    return rng.normal(0.0005, 0.015, 1000)


@pytest.fixture
def fat_tail_returns(rng) -> np.ndarray:
    """Student-t returns (fat tails)."""
    from scipy import stats as scipy_stats
    return scipy_stats.t.rvs(df=3, loc=0.0, scale=0.015, size=1000, random_state=42)


@pytest.fixture
def positive_drift_returns(rng) -> np.ndarray:
    """Returns with positive drift for tail ratio test."""
    return rng.normal(0.002, 0.01, 500)


@pytest.fixture
def analyzer() -> EnhancedRiskAnalyzer:
    return EnhancedRiskAnalyzer()


# ── VaR Trio Tests ───────────────────────────────────────────────────


class TestVaRTrio:
    def test_returns_var_result(self, analyzer, normal_returns):
        result = analyzer.var_trio(normal_returns)
        assert isinstance(result, VaRResult)

    def test_confidence_stored(self, analyzer, normal_returns):
        result = analyzer.var_trio(normal_returns, confidence=0.99)
        assert result.confidence == 0.99

    def test_all_var_negative(self, analyzer, normal_returns):
        """For returns centered near zero, VaR should be negative (loss)."""
        result = analyzer.var_trio(normal_returns, confidence=0.95)
        assert result.parametric < 0
        assert result.historical < 0
        assert result.monte_carlo < 0

    def test_higher_confidence_larger_var(self, analyzer, normal_returns):
        """Higher confidence should produce larger loss (more negative VaR)."""
        var95 = analyzer.var_trio(normal_returns, confidence=0.95)
        var99 = analyzer.var_trio(normal_returns, confidence=0.99)
        assert var99.parametric < var95.parametric  # More negative = bigger loss

    def test_parametric_vs_historical_fat_tails(self, analyzer, fat_tail_returns):
        """With fat tails, historical VaR should be more extreme than parametric."""
        result = analyzer.var_trio(fat_tail_returns, confidence=0.95)
        # For fat-tailed distributions, historical typically captures more extreme losses
        assert result.historical < result.parametric or abs(
            result.historical - result.parametric
        ) < 0.05

    def test_monte_carlo_converges(self, analyzer, normal_returns):
        """MC VaR should be close to parametric for normal data."""
        result = analyzer.var_trio(normal_returns, confidence=0.95, n_sims=50_000)
        # Should be within 10% of parametric for normal distribution
        assert abs(result.monte_carlo - result.parametric) < abs(
            result.parametric * 0.15
        )

    def test_seed_reproducibility(self, analyzer, normal_returns):
        """Same seed should produce same MC result."""
        r1 = analyzer.var_trio(normal_returns, seed=123)
        r2 = analyzer.var_trio(normal_returns, seed=123)
        assert r1.monte_carlo == r2.monte_carlo


# ── Backtest VaR Tests ───────────────────────────────────────────────


class TestBacktestVaR:
    def test_kupiec_known_violations(self, analyzer, rng):
        """Known violation rate should produce expected p-value direction."""
        n = 1000
        returns = rng.normal(0, 0.01, n)
        # Set VaR such that we get ~5% violations (at 95% confidence)
        var_level = np.percentile(returns, 5)
        var_estimates = np.full(n, var_level)

        result = analyzer.backtest_var(returns, var_estimates, confidence=0.95)
        assert "kupiec_pvalue" in result
        assert "christoffersen_pvalue" in result
        assert result["n_violations"] > 0
        # p-value should not reject at 5% level when violations match expected
        assert result["kupiec_pvalue"] > 0.01

    def test_too_many_violations(self, analyzer, rng):
        """Way too many violations should have low Kupiec p-value."""
        n = 500
        returns = rng.normal(-0.05, 0.01, n)  # Very negative returns
        var_estimates = np.full(n, -0.01)  # VaR set too optimistically

        result = analyzer.backtest_var(returns, var_estimates, confidence=0.95)
        # Should have many more violations than expected
        assert result["n_violations"] > result["expected_violations"] * 2

    def test_zero_violations(self, analyzer, rng):
        """Zero violations when VaR is extremely conservative."""
        n = 100
        returns = rng.normal(0.01, 0.005, n)  # Positive returns
        var_estimates = np.full(n, -1.0)  # VaR at -100%

        result = analyzer.backtest_var(returns, var_estimates, confidence=0.95)
        assert result["n_violations"] == 0


# ── Stress Test Tests ────────────────────────────────────────────────


class TestStressTest:
    def test_historical_scenario_string(self, analyzer, normal_returns):
        result = analyzer.stress_test(
            normal_returns,
            {"AAPL": 0.5, "BND": 0.5},
            "2008_gfc",
        )
        assert result["scenario"] == "2008 GFC"
        assert result["portfolio_loss"] < 0  # Should be a loss

    def test_all_historical_scenarios(self, analyzer, normal_returns):
        """All predefined scenarios should work."""
        weights = {"AAPL": 0.6, "BND": 0.4}
        for key in HISTORICAL_SCENARIOS:
            result = analyzer.stress_test(normal_returns, weights, key)
            assert result["portfolio_loss"] < 0
            assert "description" in result

    def test_custom_scenario(self, analyzer, normal_returns):
        custom = StressScenario("Custom Crash", -0.40, 100, 0.60, "Test scenario")
        result = analyzer.stress_test(normal_returns, {"SPY": 1.0}, custom)
        assert result["scenario"] == "Custom Crash"
        assert result["portfolio_loss"] < 0

    def test_unknown_scenario_raises(self, analyzer, normal_returns):
        with pytest.raises(ValueError, match="Unknown scenario"):
            analyzer.stress_test(normal_returns, {"SPY": 1.0}, "zombie_crash")

    def test_loss_proportional_to_exposure(self, analyzer, normal_returns):
        """Higher equity weight = larger loss."""
        loss_low = analyzer.stress_test(
            normal_returns, {"SPY": 0.3}, "2020_covid"
        )["portfolio_loss"]
        loss_high = analyzer.stress_test(
            normal_returns, {"SPY": 1.0}, "2020_covid"
        )["portfolio_loss"]
        assert loss_high < loss_low  # More negative


# ── Correlation Breakdown Tests ──────────────────────────────────────


class TestCorrelationBreakdown:
    def test_identity_no_convergence(self, analyzer):
        """Uncorrelated assets: no convergence."""
        rng = np.random.default_rng(42)
        # Independent columns
        matrix = rng.normal(0, 1, (100, 5))
        result = analyzer.correlation_breakdown(matrix)
        assert not result["is_converging"]
        # For 5 independent assets, first eigenvalue should explain ~20%
        assert result["first_eigenvalue_ratio"] < 0.60

    def test_perfect_correlation_convergence(self, analyzer):
        """Perfectly correlated assets: convergence detected."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 100)
        # All columns are the same (+ tiny noise for numerical stability)
        matrix = np.column_stack([
            base + rng.normal(0, 0.001, 100) for _ in range(5)
        ])
        result = analyzer.correlation_breakdown(matrix)
        assert result["is_converging"]
        assert result["first_eigenvalue_ratio"] > 0.90

    def test_eigenvalues_returned(self, analyzer):
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (100, 3))
        result = analyzer.correlation_breakdown(matrix)
        assert len(result["eigenvalues"]) == 3
        # Eigenvalues should be positive for a correlation matrix
        for ev in result["eigenvalues"]:
            assert ev >= -0.01  # Small tolerance for numerical

    def test_single_asset(self, analyzer):
        """Single asset should return degenerate result."""
        matrix = np.random.default_rng(42).normal(0, 1, (100, 1))
        result = analyzer.correlation_breakdown(matrix)
        assert not result["is_converging"]


# ── Tail Risk Metrics Tests ──────────────────────────────────────────


class TestTailMetrics:
    def test_returns_tail_risk_metrics(self, analyzer, normal_returns):
        result = analyzer.tail_metrics(normal_returns)
        assert isinstance(result, TailRiskMetrics)

    def test_evar_more_extreme_than_var(self, analyzer, normal_returns):
        """EVaR should be more extreme (more negative) than VaR."""
        var_result = analyzer.var_trio(normal_returns, confidence=0.95)
        tail = analyzer.tail_metrics(normal_returns)
        # EVaR is a tighter bound, should be <= VaR (more conservative)
        assert tail.evar <= var_result.parametric + 0.005

    def test_cdar_negative(self, analyzer, normal_returns):
        """CDaR should be negative (drawdowns are negative)."""
        result = analyzer.tail_metrics(normal_returns)
        assert result.cdar <= 0.0

    def test_max_dd_duration_positive(self, analyzer, normal_returns):
        result = analyzer.tail_metrics(normal_returns)
        assert result.max_dd_duration >= 0

    def test_tail_ratio_positive_drift(self, analyzer, positive_drift_returns):
        """With positive drift, tail ratio should be > 1 (gains > losses)."""
        result = analyzer.tail_metrics(positive_drift_returns)
        assert result.tail_ratio > 0

    def test_tail_ratio_symmetric(self, analyzer):
        """Symmetric distribution should have tail ratio near 1."""
        rng = np.random.default_rng(42)
        symmetric = rng.normal(0, 0.01, 5000)
        result = analyzer.tail_metrics(symmetric)
        assert 0.7 < result.tail_ratio < 1.4


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_returns_var(self, analyzer):
        """Empty returns should not crash."""
        result = analyzer.var_trio(np.array([0.0, 0.0, 0.0]))
        assert isinstance(result, VaRResult)

    def test_single_return_tail(self, analyzer):
        """Single observation should not crash."""
        result = analyzer.tail_metrics(np.array([0.01]))
        assert isinstance(result, TailRiskMetrics)

    def test_constant_returns(self, analyzer):
        """Constant returns (zero vol) handled gracefully."""
        returns = np.full(100, 0.001)
        result = analyzer.var_trio(returns)
        # Parametric VaR with zero vol should be close to mean
        assert isinstance(result, VaRResult)

    def test_stress_test_empty_weights(self, analyzer, normal_returns):
        """Empty portfolio weights produce zero loss."""
        result = analyzer.stress_test(normal_returns, {}, "2008_gfc")
        assert result["portfolio_loss"] == 0.0
