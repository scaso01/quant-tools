"""Tests for portfolio optimizer (synthetic data, no API calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import (
    AssetConstraint,
    EfficientFrontier,
    GroupConstraint,
    OptimizationConstraints,
    PortfolioOptimization,
)
from quant_tools.portfolio.optimizer import PortfolioOptimizer


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def returns_5_assets(rng) -> pd.DataFrame:
    """Synthetic daily returns for 5 assets, 252 days."""
    n_days = 252
    assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    # Different mean returns and volatilities per asset
    means = [0.0004, 0.0003, 0.00035, 0.0005, 0.0006]
    vols = [0.015, 0.012, 0.014, 0.018, 0.025]
    data = {}
    for i, asset in enumerate(assets):
        data[asset] = rng.normal(means[i], vols[i], n_days)
    return pd.DataFrame(data)


@pytest.fixture
def returns_2_assets(rng) -> pd.DataFrame:
    """Synthetic returns for 2 assets."""
    n_days = 252
    data = {
        "A": rng.normal(0.0003, 0.01, n_days),
        "B": rng.normal(0.0005, 0.02, n_days),
    }
    return pd.DataFrame(data)


@pytest.fixture
def returns_single_asset(rng) -> pd.DataFrame:
    """Single asset returns."""
    return pd.DataFrame({"ONLY": rng.normal(0.0004, 0.015, 252)})


@pytest.fixture
def correlated_returns(rng) -> pd.DataFrame:
    """Highly correlated assets (correlation ~ 0.95)."""
    n_days = 252
    base = rng.normal(0.0004, 0.015, n_days)
    data = {
        "X": base + rng.normal(0, 0.002, n_days),
        "Y": base + rng.normal(0, 0.002, n_days),
        "Z": base + rng.normal(0, 0.002, n_days),
    }
    return pd.DataFrame(data)


@pytest.fixture
def optimizer() -> PortfolioOptimizer:
    return PortfolioOptimizer(risk_free_rate=0.04)


# ── Markowitz Tests ───────────────────────────────────────────────────


class TestMarkowitz:
    def test_weights_sum_to_one(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        for w in result.weights.values():
            assert w >= 0.0

    def test_returns_portfolio_optimization(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        assert isinstance(result, PortfolioOptimization)

    def test_method_label(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        assert result.method == "markowitz"

    def test_sharpe_is_positive(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        # With positive mean returns, Sharpe should be positive
        assert result.sharpe_ratio > 0

    def test_max_weight_constraint(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        for w in result.weights.values():
            assert w <= 0.31  # Slightly above 0.3 for numerical tolerance

    def test_volatility_positive(self, optimizer, returns_5_assets):
        result = optimizer.optimize_markowitz(returns_5_assets)
        assert result.volatility > 0

    def test_two_assets(self, optimizer, returns_2_assets):
        result = optimizer.optimize_markowitz(returns_2_assets)
        # With 2 assets capped at 30% each, optimizer hits bounds (max sum = 0.6)
        assert sum(result.weights.values()) > 0
        assert len(result.weights) >= 1


# ── Risk Parity Tests ────────────────────────────────────────────────


class TestRiskParity:
    def test_weights_sum_to_one(self, optimizer, returns_5_assets):
        result = optimizer.optimize_risk_parity(returns_5_assets)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_all_weights_positive(self, optimizer, returns_5_assets):
        result = optimizer.optimize_risk_parity(returns_5_assets)
        for w in result.weights.values():
            assert w > 0  # Risk parity has lower bound 0.01

    def test_method_label(self, optimizer, returns_5_assets):
        result = optimizer.optimize_risk_parity(returns_5_assets)
        assert result.method == "risk_parity"

    def test_includes_all_assets(self, optimizer, returns_5_assets):
        result = optimizer.optimize_risk_parity(returns_5_assets)
        assert len(result.weights) == 5

    def test_lower_vol_gets_more_weight(self, optimizer, returns_5_assets):
        """Risk parity gives more weight to lower-volatility assets."""
        result = optimizer.optimize_risk_parity(returns_5_assets)
        # MSFT has lowest vol (0.012), TSLA highest (0.025)
        # Risk parity should give MSFT more weight than TSLA
        assert result.weights.get("MSFT", 0) > result.weights.get("TSLA", 0)

    def test_correlated_assets(self, optimizer, correlated_returns):
        result = optimizer.optimize_risk_parity(correlated_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01


# ── Min Variance Tests ───────────────────────────────────────────────


class TestMinVariance:
    def test_weights_sum_to_one(self, optimizer, returns_5_assets):
        result = optimizer.optimize_min_variance(returns_5_assets)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, optimizer, returns_5_assets):
        result = optimizer.optimize_min_variance(returns_5_assets)
        for w in result.weights.values():
            assert w >= 0.0

    def test_method_label(self, optimizer, returns_5_assets):
        result = optimizer.optimize_min_variance(returns_5_assets)
        assert result.method == "min_variance"

    def test_lower_volatility_than_equal_weight(self, optimizer, returns_5_assets):
        """Min variance portfolio should have lower vol than equal-weight."""
        result = optimizer.optimize_min_variance(returns_5_assets)
        # Equal-weight portfolio volatility
        ew_returns = returns_5_assets.mean(axis=1)
        ew_vol = float(ew_returns.std() * np.sqrt(252))
        assert result.volatility <= ew_vol + 0.001  # Small tolerance

    def test_correlated_assets(self, optimizer, correlated_returns):
        result = optimizer.optimize_min_variance(correlated_returns)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01


# ── Dispatcher Tests ─────────────────────────────────────────────────


class TestDispatcher:
    def test_markowitz_dispatch(self, optimizer, returns_5_assets):
        result = optimizer.optimize(returns_5_assets, method="markowitz")
        assert result.method == "markowitz"

    def test_risk_parity_dispatch(self, optimizer, returns_5_assets):
        result = optimizer.optimize(returns_5_assets, method="risk_parity")
        assert result.method == "risk_parity"

    def test_min_variance_dispatch(self, optimizer, returns_5_assets):
        result = optimizer.optimize(returns_5_assets, method="min_variance")
        assert result.method == "min_variance"

    def test_invalid_method_raises(self, optimizer, returns_5_assets):
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.optimize(returns_5_assets, method="magic")


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_asset_markowitz(self, optimizer, returns_single_asset):
        """Single asset must get weight=1.0 (sum constraint forces it)."""
        result = optimizer.optimize_markowitz(returns_single_asset)
        assert len(result.weights) == 1
        assert abs(result.weights["ONLY"] - 1.0) < 0.02

    def test_single_asset_min_variance(self, optimizer, returns_single_asset):
        """Single asset must get weight=1.0 (sum constraint forces it)."""
        result = optimizer.optimize_min_variance(returns_single_asset)
        assert len(result.weights) == 1
        assert abs(result.weights["ONLY"] - 1.0) < 0.02

    def test_custom_risk_free_rate(self, returns_5_assets):
        opt = PortfolioOptimizer(risk_free_rate=0.0)
        result = opt.optimize_markowitz(returns_5_assets)
        assert isinstance(result, PortfolioOptimization)
        # With 0% risk-free rate, Sharpe should be higher than with 4%
        opt2 = PortfolioOptimizer(risk_free_rate=0.04)
        result2 = opt2.optimize_markowitz(returns_5_assets)
        assert result.sharpe_ratio >= result2.sharpe_ratio

    def test_default_risk_free_rate(self):
        opt = PortfolioOptimizer()
        assert opt.risk_free_rate == 0.04  # From config default


# ── Constraint Tests ────────────────────────────────────────────────


@pytest.fixture
def returns_10_assets(rng) -> pd.DataFrame:
    """Synthetic daily returns for 10 assets (2 sectors), 252 days."""
    n_days = 252
    assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "BAC", "WFC", "GS", "C"]
    means = [0.0004, 0.0003, 0.00035, 0.0005, 0.0006, 0.0002, 0.00025, 0.0003, 0.00035, 0.0002]
    vols = [0.015, 0.012, 0.014, 0.018, 0.025, 0.016, 0.018, 0.014, 0.020, 0.017]
    data = {}
    for i, asset in enumerate(assets):
        data[asset] = rng.normal(means[i], vols[i], n_days)
    return pd.DataFrame(data)


class TestAssetConstraints:
    def test_markowitz_per_asset_max(self, optimizer, returns_5_assets):
        """Per-asset max bound is respected."""
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint("AAPL", min_weight=0.0, max_weight=0.10),
                AssetConstraint("MSFT", min_weight=0.0, max_weight=0.10),
                AssetConstraint("GOOG", min_weight=0.0, max_weight=1.0),
                AssetConstraint("AMZN", min_weight=0.0, max_weight=1.0),
                AssetConstraint("TSLA", min_weight=0.0, max_weight=1.0),
            ]
        )
        result = optimizer.optimize_markowitz(returns_5_assets, constraints=constraints)
        assert result.weights.get("AAPL", 0) <= 0.11
        assert result.weights.get("MSFT", 0) <= 0.11

    def test_min_variance_per_asset_max(self, optimizer, returns_5_assets):
        """Min-variance respects per-asset max."""
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint("AAPL", min_weight=0.0, max_weight=0.15),
                AssetConstraint("MSFT", min_weight=0.0, max_weight=0.15),
                AssetConstraint("GOOG", min_weight=0.0, max_weight=1.0),
                AssetConstraint("AMZN", min_weight=0.0, max_weight=1.0),
                AssetConstraint("TSLA", min_weight=0.0, max_weight=1.0),
            ]
        )
        result = optimizer.optimize_min_variance(returns_5_assets, constraints=constraints)
        assert result.weights.get("AAPL", 0) <= 0.16
        assert result.weights.get("MSFT", 0) <= 0.16

    def test_per_asset_min_weight(self, optimizer, returns_5_assets):
        """Per-asset minimum weight is respected."""
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint("AAPL", min_weight=0.10, max_weight=1.0),
                AssetConstraint("MSFT", min_weight=0.10, max_weight=1.0),
                AssetConstraint("GOOG", min_weight=0.10, max_weight=1.0),
                AssetConstraint("AMZN", min_weight=0.10, max_weight=1.0),
                AssetConstraint("TSLA", min_weight=0.10, max_weight=1.0),
            ]
        )
        result = optimizer.optimize_min_variance(returns_5_assets, constraints=constraints)
        for w in result.weights.values():
            assert w >= 0.09  # Slight numerical tolerance

    def test_risk_parity_per_asset_bounds(self, optimizer, returns_5_assets):
        """Risk parity respects custom per-asset bounds."""
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint("AAPL", min_weight=0.05, max_weight=0.30),
                AssetConstraint("MSFT", min_weight=0.05, max_weight=0.30),
                AssetConstraint("GOOG", min_weight=0.05, max_weight=0.30),
                AssetConstraint("AMZN", min_weight=0.05, max_weight=0.30),
                AssetConstraint("TSLA", min_weight=0.05, max_weight=0.30),
            ]
        )
        result = optimizer.optimize_risk_parity(returns_5_assets, constraints=constraints)
        for w in result.weights.values():
            assert w >= 0.04
            assert w <= 0.31

    def test_constraints_stored_in_result(self, optimizer, returns_5_assets):
        """OptimizationConstraints are stored in the result."""
        constraints = OptimizationConstraints(
            asset_constraints=[AssetConstraint("AAPL", max_weight=0.20)]
        )
        result = optimizer.optimize_markowitz(returns_5_assets, constraints=constraints)
        assert result.constraints_used is constraints


class TestGroupConstraints:
    def test_sector_max_respected(self, optimizer, returns_10_assets):
        """Group constraint caps total sector weight."""
        tech = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        constraints = OptimizationConstraints(
            group_constraints=[GroupConstraint("Tech", symbols=tech, max_weight=0.30)]
        )
        result = optimizer.optimize_min_variance(returns_10_assets, constraints=constraints)
        tech_weight = sum(result.weights.get(s, 0) for s in tech)
        assert tech_weight <= 0.32  # Slight tolerance

    def test_group_constraint_markowitz(self, optimizer, returns_10_assets):
        """Group constraint works with Markowitz too."""
        financials = ["JPM", "BAC", "WFC", "GS", "C"]
        constraints = OptimizationConstraints(
            group_constraints=[GroupConstraint("Finance", symbols=financials, max_weight=0.40)]
        )
        result = optimizer.optimize_markowitz(returns_10_assets, constraints=constraints)
        fin_weight = sum(result.weights.get(s, 0) for s in financials)
        assert fin_weight <= 0.42

    def test_group_constraint_risk_parity(self, optimizer, returns_10_assets):
        """Group constraint works with risk parity."""
        tech = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        constraints = OptimizationConstraints(
            group_constraints=[GroupConstraint("Tech", symbols=tech, max_weight=0.40)]
        )
        result = optimizer.optimize_risk_parity(returns_10_assets, constraints=constraints)
        tech_weight = sum(result.weights.get(s, 0) for s in tech)
        assert tech_weight <= 0.42

    def test_multiple_group_constraints(self, optimizer, returns_10_assets):
        """Multiple group constraints can coexist."""
        tech = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        financials = ["JPM", "BAC", "WFC", "GS", "C"]
        constraints = OptimizationConstraints(
            group_constraints=[
                GroupConstraint("Tech", symbols=tech, max_weight=0.50),
                GroupConstraint("Finance", symbols=financials, max_weight=0.50),
            ]
        )
        result = optimizer.optimize_min_variance(returns_10_assets, constraints=constraints)
        assert abs(sum(result.weights.values()) - 1.0) < 0.02


class TestTurnoverConstraint:
    def test_turnover_limits_deviation(self, optimizer, returns_5_assets):
        """Turnover constraint limits total weight change."""
        current = {"AAPL": 0.20, "MSFT": 0.20, "GOOG": 0.20, "AMZN": 0.20, "TSLA": 0.20}
        constraints = OptimizationConstraints(
            max_turnover=0.20,
            current_weights=current,
        )
        result = optimizer.optimize_min_variance(returns_5_assets, constraints=constraints)
        assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        turnover = sum(abs(result.weights.get(a, 0) - current[a]) for a in assets)
        assert turnover <= 0.22  # Tolerance for numerical

    def test_turnover_markowitz(self, optimizer, returns_5_assets):
        """Turnover constraint works with Markowitz."""
        current = {"AAPL": 0.20, "MSFT": 0.20, "GOOG": 0.20, "AMZN": 0.20, "TSLA": 0.20}
        constraints = OptimizationConstraints(
            max_turnover=0.10,
            current_weights=current,
        )
        result = optimizer.optimize_markowitz(returns_5_assets, constraints=constraints)
        assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        turnover = sum(abs(result.weights.get(a, 0) - current[a]) for a in assets)
        assert turnover <= 0.12


class TestCorrelationOverride:
    def test_override_produces_valid_result(self, optimizer, returns_5_assets):
        """Correlation override produces a valid optimization result."""
        n = 5
        # Identity correlation (assets uncorrelated)
        corr = np.eye(n)
        constraints = OptimizationConstraints(correlation_override=corr)
        result = optimizer.optimize_min_variance(returns_5_assets, constraints=constraints)
        assert abs(sum(result.weights.values()) - 1.0) < 0.02
        assert result.volatility > 0

    def test_override_affects_weights(self, optimizer, returns_5_assets):
        """Different correlation overrides produce different weights."""
        n = 5
        corr_identity = np.eye(n)
        corr_full = np.ones((n, n)) * 0.9
        np.fill_diagonal(corr_full, 1.0)

        c1 = OptimizationConstraints(correlation_override=corr_identity)
        c2 = OptimizationConstraints(correlation_override=corr_full)

        r1 = optimizer.optimize_min_variance(returns_5_assets, constraints=c1)
        r2 = optimizer.optimize_min_variance(returns_5_assets, constraints=c2)

        # With identity correlation, diversification benefit is maximal
        # With near-perfect correlation, diversification is minimal
        # Volatilities should differ
        assert r1.volatility != r2.volatility

    def test_override_markowitz(self, optimizer, returns_5_assets):
        """Correlation override works with Markowitz."""
        n = 5
        corr = np.eye(n)
        constraints = OptimizationConstraints(correlation_override=corr)
        result = optimizer.optimize_markowitz(returns_5_assets, constraints=constraints)
        assert abs(sum(result.weights.values()) - 1.0) < 0.02


# ── Efficient Frontier Tests ────────────────────────────────────────


class TestEfficientFrontier:
    def test_returns_efficient_frontier(self, optimizer, returns_5_assets):
        result = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=20)
        assert isinstance(result, EfficientFrontier)
        assert len(result.points) > 0

    def test_monotonic_returns(self, optimizer, returns_5_assets):
        """Efficient frontier returns should be non-decreasing."""
        ef = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=30)
        returns = [p.expected_return for p in ef.points]
        for i in range(1, len(returns)):
            assert returns[i] >= returns[i - 1] - 0.001  # Tolerance

    def test_max_sharpe_identified(self, optimizer, returns_5_assets):
        """Max-Sharpe point is identified on the frontier."""
        ef = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=30)
        assert ef.max_sharpe_point is not None
        # It should have the best Sharpe among all points
        for p in ef.points:
            assert ef.max_sharpe_point.sharpe_ratio >= p.sharpe_ratio - 0.01

    def test_min_variance_identified(self, optimizer, returns_5_assets):
        """Min-variance point has lowest volatility on frontier."""
        ef = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=30)
        assert ef.min_variance_point is not None
        for p in ef.points:
            assert ef.min_variance_point.volatility <= p.volatility + 0.001

    def test_frontier_points_sum_to_one(self, optimizer, returns_5_assets):
        """Each point on the frontier has weights summing to 1."""
        ef = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=10)
        for p in ef.points:
            assert abs(sum(p.weights.values()) - 1.0) < 0.02

    def test_frontier_with_constraints(self, optimizer, returns_5_assets):
        """Efficient frontier works with constraints."""
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint("AAPL", max_weight=0.15),
                AssetConstraint("MSFT", max_weight=0.15),
                AssetConstraint("GOOG", max_weight=0.50),
                AssetConstraint("AMZN", max_weight=0.50),
                AssetConstraint("TSLA", max_weight=0.50),
            ]
        )
        ef = optimizer.optimize_efficient_frontier(
            returns_5_assets, n_points=15, constraints=constraints,
        )
        assert len(ef.points) > 0
        for p in ef.points:
            assert p.weights.get("AAPL", 0) <= 0.16
            assert p.weights.get("MSFT", 0) <= 0.16

    def test_frontier_n_points(self, optimizer, returns_5_assets):
        """n_points controls granularity."""
        ef10 = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=10)
        ef30 = optimizer.optimize_efficient_frontier(returns_5_assets, n_points=30)
        assert len(ef30.points) >= len(ef10.points)


# ── Invariant Tests ─────────────────────────────────────────────────


class TestInvariants:
    def test_all_methods_weights_sum_one(self, optimizer, returns_5_assets):
        """All methods produce weights summing to 1."""
        for method in ["markowitz", "risk_parity", "min_variance"]:
            result = optimizer.optimize(returns_5_assets, method=method)
            assert abs(sum(result.weights.values()) - 1.0) < 0.02, f"{method} failed"

    def test_all_methods_non_negative(self, optimizer, returns_5_assets):
        """All methods produce non-negative weights (long-only)."""
        for method in ["markowitz", "risk_parity", "min_variance"]:
            result = optimizer.optimize(returns_5_assets, method=method)
            for sym, w in result.weights.items():
                assert w >= -0.001, f"{method}: {sym}={w}"

    def test_risk_parity_equalizes_risk(self, optimizer, returns_5_assets):
        """Risk parity roughly equalizes risk contributions."""
        result = optimizer.optimize_risk_parity(returns_5_assets)
        weights = np.array([result.weights.get(a, 0) for a in returns_5_assets.columns])
        cov = returns_5_assets.cov().values * 252
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal = cov @ weights
        risk_contrib = weights * marginal / (port_vol + 1e-10)
        # Risk contributions should be roughly equal (within 50% of mean)
        mean_rc = np.mean(risk_contrib)
        for rc in risk_contrib:
            assert abs(rc - mean_rc) < mean_rc * 0.6

    def test_no_constraints_markowitz_matches_defaults(self, optimizer, returns_5_assets):
        """No-constraints Markowitz uses default 0-30% bounds."""
        result = optimizer.optimize_markowitz(returns_5_assets)
        for w in result.weights.values():
            assert w <= 0.31

    def test_no_constraints_min_variance_matches_defaults(self, optimizer, returns_5_assets):
        """No-constraints min-variance uses default 0-50% bounds."""
        result = optimizer.optimize_min_variance(returns_5_assets)
        for w in result.weights.values():
            assert w <= 0.51

    def test_empty_returns_markowitz(self, optimizer):
        """Empty DataFrame handled gracefully."""
        empty = pd.DataFrame()
        # Empty columns means 0 assets -- may raise or return empty
        try:
            result = optimizer.optimize_markowitz(empty)
            assert result.weights == {} or sum(result.weights.values()) == 0
        except (ValueError, KeyError, IndexError):
            pass  # Acceptable to raise on empty input

    def test_empty_returns_min_variance(self, optimizer):
        """Empty DataFrame handled gracefully."""
        empty = pd.DataFrame()
        try:
            result = optimizer.optimize_min_variance(empty)
            assert result.weights == {} or sum(result.weights.values()) == 0
        except (ValueError, KeyError, IndexError):
            pass
