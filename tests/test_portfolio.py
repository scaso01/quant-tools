"""Tests for portfolio analyzer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.errors import DataError
from quant_tools.models import PortfolioAnalysis, PortfolioHolding
from quant_tools.portfolio.analyzer import (
    analyze_portfolio,
    calculate_correlation_matrix,
    calculate_portfolio_metrics,
    calculate_sector_exposure,
)


# =============================================================================
# Test calculate_correlation_matrix
# =============================================================================


class TestCorrelationMatrix:
    """Test correlation matrix calculation."""

    def test_perfect_positive_correlation(self):
        """Identical arrays should have correlation 1.0."""
        arr = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        returns_dict = {"A": arr, "B": arr.copy()}
        matrix = calculate_correlation_matrix(returns_dict)
        assert matrix["A"]["A"] == pytest.approx(1.0)
        assert matrix["A"]["B"] == pytest.approx(1.0)
        assert matrix["B"]["A"] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Negated arrays should have correlation -1.0."""
        arr = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        returns_dict = {"A": arr, "B": -arr}
        matrix = calculate_correlation_matrix(returns_dict)
        assert matrix["A"]["B"] == pytest.approx(-1.0)

    def test_diagonal_is_one(self):
        """Self-correlation should always be 1.0."""
        np.random.seed(99)
        returns_dict = {
            "X": np.random.normal(0, 0.01, 100),
            "Y": np.random.normal(0, 0.02, 100),
            "Z": np.random.normal(0, 0.015, 100),
        }
        matrix = calculate_correlation_matrix(returns_dict)
        for sym in returns_dict:
            assert matrix[sym][sym] == pytest.approx(1.0)

    def test_single_asset(self):
        """Single asset should produce 1x1 matrix with 1.0."""
        returns_dict = {"ONLY": np.array([0.01, 0.02, -0.01])}
        matrix = calculate_correlation_matrix(returns_dict)
        assert matrix["ONLY"]["ONLY"] == pytest.approx(1.0)

    def test_short_arrays_return_zero(self):
        """Arrays with fewer than 2 points return 0.0 correlation."""
        returns_dict = {"A": np.array([0.01]), "B": np.array([0.02])}
        matrix = calculate_correlation_matrix(returns_dict)
        assert matrix["A"]["B"] == 0.0


# =============================================================================
# Test calculate_sector_exposure
# =============================================================================


class TestSectorExposure:
    """Test sector exposure calculation."""

    def test_sector_aggregation(self):
        """Holdings in same sector should have weights summed."""
        holdings = {"AAPL": 0.3, "MSFT": 0.2, "BND": 0.5}
        mock_infos = {
            "AAPL": {"sector": "Technology"},
            "MSFT": {"sector": "Technology"},
            "BND": {"sector": "Fixed Income"},
        }

        with patch(
            "quant_tools.portfolio.analyzer.fetch_ticker_info",
            side_effect=lambda s: mock_infos[s],
        ):
            exposure = calculate_sector_exposure(holdings)

        assert exposure["Technology"] == pytest.approx(0.5)
        assert exposure["Fixed Income"] == pytest.approx(0.5)

    def test_unknown_sector_on_error(self):
        """Failed ticker info should map to Unknown sector.

        DataError, not a bare Exception: that is what fetch_ticker_info
        documents itself as raising, and the handler is now narrow enough that
        an unexpected error surfaces instead of quietly becoming a slice.
        """
        holdings = {"FAKE": 1.0}

        with patch(
            "quant_tools.portfolio.analyzer.fetch_ticker_info",
            side_effect=DataError("API error"),
        ):
            exposure = calculate_sector_exposure(holdings)

        assert "Unknown" in exposure
        assert exposure["Unknown"] == pytest.approx(1.0)


# =============================================================================
# Test calculate_portfolio_metrics
# =============================================================================


class TestPortfolioMetrics:
    """Test portfolio-level metric calculation."""

    def test_basic_metrics(self):
        """Verify basic metric computation on known data."""
        np.random.seed(42)
        n = 252
        returns = np.random.normal(0.0005, 0.01, (n, 2))
        weights = np.array([0.6, 0.4])

        metrics = calculate_portfolio_metrics(weights, returns, risk_free_rate=0.04)

        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["volatility"] > 0
        assert metrics["max_drawdown"] <= 0

    def test_zero_volatility(self):
        """Zero-vol returns should produce sharpe of 0."""
        n = 100
        returns = np.zeros((n, 1))
        weights = np.array([1.0])

        metrics = calculate_portfolio_metrics(weights, returns, risk_free_rate=0.0)

        assert metrics["volatility"] == pytest.approx(0.0)
        assert metrics["sharpe_ratio"] == pytest.approx(0.0)

    def test_all_same_returns(self):
        """Identical daily returns across assets should give predictable results."""
        n = 100
        daily_ret = 0.001
        returns = np.full((n, 3), daily_ret)
        weights = np.array([0.5, 0.3, 0.2])

        metrics = calculate_portfolio_metrics(weights, returns, risk_free_rate=0.0)

        expected_total = (1 + daily_ret) ** n - 1
        assert metrics["total_return"] == pytest.approx(expected_total, rel=1e-6)


# =============================================================================
# Test analyze_portfolio (integration, mocked market data)
# =============================================================================


class TestAnalyzePortfolio:
    """Test full analyze_portfolio function with mocked data."""

    def test_analyze_with_mocked_data(self, sample_multi_prices):
        """Full analysis with mocked fetch_prices and fetch_ticker_info."""
        holdings = {"AAPL": 0.4, "MSFT": 0.3, "BND": 0.3}

        mock_infos = {
            "AAPL": {"sector": "Technology"},
            "MSFT": {"sector": "Technology"},
            "BND": {"sector": "Fixed Income"},
        }

        with (
            patch(
                "quant_tools.portfolio.analyzer.fetch_prices",
                side_effect=lambda symbol, **kw: sample_multi_prices[symbol],
            ),
            patch(
                "quant_tools.portfolio.analyzer.fetch_ticker_info",
                side_effect=lambda s: mock_infos[s],
            ),
        ):
            result = analyze_portfolio(holdings, period="1y")

        assert isinstance(result, PortfolioAnalysis)
        assert len(result.holdings) == 3
        assert result.volatility > 0
        assert result.max_drawdown <= 0

        # Check holdings match input
        symbols = {h.symbol for h in result.holdings}
        assert symbols == {"AAPL", "MSFT", "BND"}

        # Weights preserved
        for h in result.holdings:
            assert h.weight == holdings[h.symbol]

        # Correlation matrix populated
        assert "AAPL" in result.correlation_matrix
        assert "MSFT" in result.correlation_matrix["AAPL"]

        # Sector exposure populated
        assert "Technology" in result.sector_exposure

    def test_analyze_single_holding(self, sample_multi_prices):
        """Single holding should still produce valid analysis."""
        holdings = {"AAPL": 1.0}

        with (
            patch(
                "quant_tools.portfolio.analyzer.fetch_prices",
                side_effect=lambda symbol, **kw: sample_multi_prices[symbol],
            ),
            patch(
                "quant_tools.portfolio.analyzer.fetch_ticker_info",
                return_value={"sector": "Technology"},
            ),
        ):
            result = analyze_portfolio(holdings)

        assert len(result.holdings) == 1
        assert result.holdings[0].weight == 1.0
        assert result.correlation_matrix["AAPL"]["AAPL"] == pytest.approx(1.0)

    def test_analyze_preserves_holding_order(self, sample_multi_prices):
        """Holdings should be returned in the same order as input."""
        holdings = {"BND": 0.5, "AAPL": 0.3, "MSFT": 0.2}

        with (
            patch(
                "quant_tools.portfolio.analyzer.fetch_prices",
                side_effect=lambda symbol, **kw: sample_multi_prices[symbol],
            ),
            patch(
                "quant_tools.portfolio.analyzer.fetch_ticker_info",
                return_value={"sector": "Unknown"},
            ),
        ):
            result = analyze_portfolio(holdings)

        result_symbols = [h.symbol for h in result.holdings]
        assert result_symbols == ["BND", "AAPL", "MSFT"]
