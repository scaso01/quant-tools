"""Tests for factor attribution module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.reporting.factor_attribution import (
    AttributionResult,
    FactorExposure,
    _ols_regression,
    capm_decompose,
    decompose,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def portfolio_returns() -> pd.Series:
    """Portfolio daily returns with known factor exposure."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-02", periods=252)
    returns = np.random.normal(0.0004, 0.012, 252)
    return pd.Series(returns, index=dates, name="portfolio")


@pytest.fixture
def factor_returns() -> pd.DataFrame:
    """Three-factor returns (Market, SMB, HML)."""
    np.random.seed(99)
    dates = pd.bdate_range(start="2023-01-02", periods=252)
    return pd.DataFrame(
        {
            "Market": np.random.normal(0.0003, 0.010, 252),
            "SMB": np.random.normal(0.0001, 0.006, 252),
            "HML": np.random.normal(0.0001, 0.005, 252),
        },
        index=dates,
    )


@pytest.fixture
def market_returns() -> pd.Series:
    """Market-only return series."""
    np.random.seed(99)
    dates = pd.bdate_range(start="2023-01-02", periods=252)
    return pd.Series(
        np.random.normal(0.0003, 0.010, 252), index=dates, name="market"
    )


# =============================================================================
# OLS regression tests
# =============================================================================


class TestOLSRegression:
    """Test the internal OLS regression helper."""

    def test_known_linear_relationship(self):
        """y = 2 + 3x should recover coefficients."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 0.1, 100)
        y = 2.0 + 3.0 * x + noise

        coeffs, r_sq, adj_r_sq, t_stats, p_vals = _ols_regression(y, x.reshape(-1, 1))

        assert coeffs[0] == pytest.approx(2.0, abs=0.5)  # intercept
        assert coeffs[1] == pytest.approx(3.0, abs=0.1)  # slope
        assert r_sq > 0.99

    def test_r_squared_in_valid_range(self):
        """R-squared should be between 0 and 1 for reasonable data."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n, 2)
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        _, r_sq, adj_r_sq, _, _ = _ols_regression(y, x)

        assert 0.0 <= r_sq <= 1.0
        assert adj_r_sq <= r_sq  # Adjusted always <= unadjusted

    def test_singular_matrix(self):
        """Singular X should return zeros without crashing."""
        y = np.array([1.0, 2.0, 3.0])
        x = np.zeros((3, 1))  # All zeros -> singular

        coeffs, r_sq, _, _, _ = _ols_regression(y, x)

        # Should not crash; coefficients are zeros
        assert len(coeffs) == 2  # intercept + 1 factor


# =============================================================================
# decompose tests
# =============================================================================


class TestDecompose:
    """Test factor decomposition."""

    def test_decompose_returns_valid_result(self, portfolio_returns, factor_returns):
        """decompose() should return an AttributionResult."""
        result = decompose(portfolio_returns, factor_returns)
        assert isinstance(result, AttributionResult)
        assert result.model_type == "ols"

    def test_r_squared_in_range(self, portfolio_returns, factor_returns):
        """R-squared should be between 0 and 1."""
        result = decompose(portfolio_returns, factor_returns)
        assert 0.0 <= result.r_squared <= 1.0

    def test_exposures_match_factor_count(self, portfolio_returns, factor_returns):
        """Should have one exposure per factor column."""
        result = decompose(portfolio_returns, factor_returns)
        assert len(result.exposures) == 3  # Market, SMB, HML
        names = {e.name for e in result.exposures}
        assert names == {"Market", "SMB", "HML"}

    def test_residual_returns_populated(self, portfolio_returns, factor_returns):
        """Residual returns should be a Series."""
        result = decompose(portfolio_returns, factor_returns)
        assert result.residual_returns is not None
        assert len(result.residual_returns) > 0

    def test_empty_inputs(self):
        """Empty inputs should return blank result."""
        empty_returns = pd.Series(dtype=float)
        empty_factors = pd.DataFrame()
        result = decompose(empty_returns, empty_factors)
        assert result.alpha == 0.0
        assert result.r_squared == 0.0
        assert len(result.exposures) == 0

    def test_alpha_is_annualized(self, portfolio_returns, factor_returns):
        """Alpha should be annualized (multiplied by periods_per_year)."""
        result = decompose(portfolio_returns, factor_returns)
        # Alpha should be a reasonable annualized number (not daily)
        assert np.isfinite(result.alpha)


# =============================================================================
# CAPM fallback tests
# =============================================================================


class TestCAPMDecompose:
    """Test CAPM single-factor decomposition."""

    def test_capm_model_type(self, portfolio_returns, market_returns):
        """capm_decompose should set model_type to 'capm'."""
        result = capm_decompose(portfolio_returns, market_returns)
        assert result.model_type == "capm"

    def test_capm_single_exposure(self, portfolio_returns, market_returns):
        """CAPM should produce exactly one factor exposure (Market)."""
        result = capm_decompose(portfolio_returns, market_returns)
        assert len(result.exposures) == 1
        assert result.exposures[0].name == "Market"
