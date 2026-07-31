"""Tests for tearsheet reporting module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.reporting.tearsheet import (
    TearsheetGenerator,
    compute_metrics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def daily_returns() -> pd.Series:
    """Generate a realistic daily return series (252 days)."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-02", periods=252)
    returns = np.random.normal(0.0004, 0.012, 252)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def benchmark_returns() -> pd.Series:
    """Generate benchmark returns correlated with the portfolio."""
    np.random.seed(99)
    dates = pd.bdate_range(start="2023-01-02", periods=252)
    returns = np.random.normal(0.0003, 0.010, 252)
    return pd.Series(returns, index=dates, name="benchmark")


# =============================================================================
# compute_metrics tests
# =============================================================================


class TestComputeMetrics:
    """Test tearsheet metric computation."""

    def test_total_return_positive_for_positive_drift(self, daily_returns):
        """Positive mean returns should yield positive total return."""
        data = compute_metrics(daily_returns)
        # With seed 42 and positive drift, total return should be > 0
        assert data.total_return > 0

    def test_cagr_computed(self, daily_returns):
        """CAGR should be a finite number."""
        data = compute_metrics(daily_returns)
        assert np.isfinite(data.cagr)

    def test_sharpe_positive_for_positive_returns(self, daily_returns):
        """Sharpe should be positive when mean excess return is positive."""
        data = compute_metrics(daily_returns, risk_free_rate=0.0)
        assert data.sharpe_ratio > 0

    def test_sortino_computed(self, daily_returns):
        """Sortino ratio should be finite and positive for positive drift."""
        data = compute_metrics(daily_returns)
        assert np.isfinite(data.sortino_ratio)
        assert data.sortino_ratio > 0

    def test_max_drawdown_in_valid_range(self, daily_returns):
        """Max drawdown should be between 0 and 1."""
        data = compute_metrics(daily_returns)
        assert 0.0 <= data.max_drawdown <= 1.0

    def test_volatility_positive(self, daily_returns):
        """Annualized volatility should be positive."""
        data = compute_metrics(daily_returns)
        assert data.volatility > 0

    def test_monthly_returns_populated(self, daily_returns):
        """Monthly returns DataFrame should be created."""
        data = compute_metrics(daily_returns)
        assert data.monthly_returns is not None
        assert not data.monthly_returns.empty

    def test_yearly_returns_populated(self, daily_returns):
        """Yearly returns Series should be created."""
        data = compute_metrics(daily_returns)
        assert data.yearly_returns is not None
        assert not data.yearly_returns.empty

    def test_benchmark_metrics(self, daily_returns, benchmark_returns):
        """Benchmark metrics should include alpha, beta, etc."""
        data = compute_metrics(daily_returns, benchmark_returns=benchmark_returns)
        assert "alpha" in data.benchmark_metrics
        assert "beta" in data.benchmark_metrics
        assert "tracking_error" in data.benchmark_metrics
        assert "information_ratio" in data.benchmark_metrics

    def test_empty_returns(self):
        """Empty series should return zero metrics."""
        empty = pd.Series(dtype=float)
        data = compute_metrics(empty)
        assert data.total_return == 0.0
        assert data.cagr == 0.0
        assert data.sharpe_ratio == 0.0

    def test_calmar_ratio(self, daily_returns):
        """Calmar ratio should be CAGR / max_drawdown."""
        data = compute_metrics(daily_returns)
        if data.max_drawdown > 0:
            expected = data.cagr / data.max_drawdown
            assert data.calmar_ratio == pytest.approx(expected, rel=1e-6)


# =============================================================================
# TearsheetGenerator tests
# =============================================================================


class TestTearsheetGenerator:
    """Test HTML tearsheet generation."""

    def test_generate_html_contains_title(self, daily_returns):
        """HTML should contain the tearsheet title."""
        data = compute_metrics(daily_returns)
        gen = TearsheetGenerator(title="Test Portfolio")
        html = gen.generate_html(data)
        assert "Test Portfolio" in html

    def test_generate_html_has_summary_section(self, daily_returns):
        """HTML should include the summary section."""
        data = compute_metrics(daily_returns)
        gen = TearsheetGenerator()
        html = gen.generate_html(data)
        assert "Summary" in html
        assert "Total Return" in html
        assert "Sharpe Ratio" in html

    def test_generate_html_has_drawdown_section(self, daily_returns):
        """HTML should include drawdown section."""
        data = compute_metrics(daily_returns)
        gen = TearsheetGenerator()
        html = gen.generate_html(data)
        assert "Drawdown" in html
        assert "Maximum Drawdown" in html

    def test_generate_html_has_returns_section(self, daily_returns):
        """HTML should include returns section."""
        data = compute_metrics(daily_returns)
        gen = TearsheetGenerator()
        html = gen.generate_html(data)
        assert "Returns" in html

    def test_generate_html_benchmark_section(self, daily_returns, benchmark_returns):
        """Benchmark section should appear when benchmark metrics exist."""
        data = compute_metrics(daily_returns, benchmark_returns=benchmark_returns)
        gen = TearsheetGenerator()
        html = gen.generate_html(data)
        assert "Benchmark Comparison" in html
        assert "Alpha" in html
        assert "Beta" in html

    def test_generate_html_is_valid_structure(self, daily_returns):
        """HTML should have proper open/close tags."""
        data = compute_metrics(daily_returns)
        gen = TearsheetGenerator()
        html = gen.generate_html(data)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<body>" in html
        assert "</body>" in html
