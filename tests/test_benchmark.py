"""Tests for benchmark framework."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import BenchmarkDefinition, BenchmarkMetrics
from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS, BenchmarkAnalyzer


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_download(returns: np.ndarray) -> pd.DataFrame:
    """Build a mock yfinance download result from daily returns."""
    prices = np.cumprod(np.concatenate([[100.0], 1.0 + returns]))
    df = pd.DataFrame({"Close": prices})
    return df


def _random_returns(n: int = 252, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.01, n)


# =============================================================================
# STANDARD_BENCHMARKS
# =============================================================================


class TestStandardBenchmarks:
    """Test STANDARD_BENCHMARKS dict."""

    def test_has_expected_keys(self):
        expected = {"SPY", "AGG", "60/40", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "GLD"}
        assert set(STANDARD_BENCHMARKS.keys()) == expected

    def test_all_values_are_benchmark_definitions(self):
        for key, val in STANDARD_BENCHMARKS.items():
            assert isinstance(val, BenchmarkDefinition), f"{key} is not BenchmarkDefinition"

    def test_composite_benchmark_has_weights(self):
        b = STANDARD_BENCHMARKS["60/40"]
        assert b.weights == {"SPY": 0.6, "AGG": 0.4}
        assert b.symbol is None

    def test_single_symbol_benchmarks_have_symbol(self):
        for key in ("SPY", "AGG", "QQQ", "IWM"):
            assert STANDARD_BENCHMARKS[key].symbol == key


# =============================================================================
# BenchmarkAnalyzer.fetch_benchmark_returns
# =============================================================================


class TestFetchBenchmarkReturns:
    """Test benchmark return fetching (mocked yfinance)."""

    def test_single_symbol_benchmark(self):
        """Single-symbol benchmark fetches from yfinance."""
        returns = _random_returns(50, seed=1)
        mock_df = _make_mock_download(returns)
        analyzer = BenchmarkAnalyzer()

        with patch("quant_tools.portfolio.benchmark.yf.download", return_value=mock_df):
            result = analyzer.fetch_benchmark_returns(
                BenchmarkDefinition(name="Test", symbol="SPY"),
            )

        assert len(result) == 50
        assert result.dtype == np.float64

    def test_composite_benchmark_combines_weights(self):
        """Composite benchmark combines weighted component returns."""
        ret_a = _random_returns(50, seed=10)
        ret_b = _random_returns(50, seed=20)
        df_a = _make_mock_download(ret_a)
        df_b = _make_mock_download(ret_b)

        def mock_download(symbol, **kwargs):
            return df_a if symbol == "SPY" else df_b

        analyzer = BenchmarkAnalyzer()
        bench = BenchmarkDefinition(name="60/40", weights={"SPY": 0.6, "AGG": 0.4})

        with patch("quant_tools.portfolio.benchmark.yf.download", side_effect=mock_download):
            result = analyzer.fetch_benchmark_returns(bench)

        # Should be weighted combination
        expected = 0.6 * ret_a + 0.4 * ret_b
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_custom_returns_passthrough(self):
        """Custom returns array is returned directly."""
        custom = np.array([0.01, -0.02, 0.005], dtype=np.float64)
        bench = BenchmarkDefinition(name="Custom", returns=custom)
        analyzer = BenchmarkAnalyzer()

        result = analyzer.fetch_benchmark_returns(bench)
        np.testing.assert_array_equal(result, custom)

    def test_empty_benchmark_returns_empty(self):
        """Benchmark with no symbol/weights/returns returns empty array."""
        bench = BenchmarkDefinition(name="Empty")
        analyzer = BenchmarkAnalyzer()
        result = analyzer.fetch_benchmark_returns(bench)
        assert len(result) == 0


# =============================================================================
# BenchmarkAnalyzer.calculate_benchmark_metrics
# =============================================================================


class TestCalculateBenchmarkMetrics:
    """Test benchmark-relative metric calculation."""

    def test_identity_returns_beta_one(self):
        """Identical returns should give beta=1, alpha~0, TE=0."""
        returns = _random_returns(252, seed=99)
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(returns, returns, "Self")

        assert metrics.beta == pytest.approx(1.0, abs=1e-10)
        assert metrics.alpha == pytest.approx(0.0, abs=1e-6)
        assert metrics.tracking_error == pytest.approx(0.0, abs=1e-10)
        assert metrics.correlation == pytest.approx(1.0, abs=1e-10)

    def test_outperformance_positive_alpha(self):
        """Portfolio that consistently outperforms should have positive alpha."""
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0003, 0.01, 252)
        # Add consistent outperformance
        portfolio = bench + 0.001
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(portfolio, bench, "Test Bench")

        assert metrics.alpha > 0.0

    def test_returns_valid_benchmark_metrics(self):
        """Result should be a BenchmarkMetrics dataclass."""
        returns = _random_returns(100, seed=7)
        bench = _random_returns(100, seed=8)
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(returns, bench, "TestBM")

        assert isinstance(metrics, BenchmarkMetrics)
        assert metrics.benchmark_name == "TestBM"

    def test_correlation_bounded(self):
        """Correlation must be between -1 and 1."""
        rng = np.random.default_rng(55)
        for seed in range(5):
            port = rng.normal(0.0, 0.01, 200)
            bench = rng.normal(0.0, 0.01, 200)
            analyzer = BenchmarkAnalyzer()
            metrics = analyzer.calculate_benchmark_metrics(port, bench)
            assert -1.0 <= metrics.correlation <= 1.0

    def test_all_metrics_finite(self):
        """No NaN or inf in any metric field."""
        port = _random_returns(252, seed=33)
        bench = _random_returns(252, seed=44)
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(port, bench, "FiniteCheck")

        assert np.isfinite(metrics.beta)
        assert np.isfinite(metrics.alpha)
        assert np.isfinite(metrics.tracking_error)
        assert np.isfinite(metrics.information_ratio)
        assert np.isfinite(metrics.treynor_ratio)
        assert np.isfinite(metrics.correlation)
        assert np.isfinite(metrics.active_return)

    def test_negative_beta_for_inverse(self):
        """Inversely related returns should give negative beta."""
        rng = np.random.default_rng(77)
        bench = rng.normal(0.0005, 0.01, 252)
        portfolio = -bench  # Perfectly inverse
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(portfolio, bench)

        assert metrics.beta < 0.0

    def test_benchmark_name_stored(self):
        """Benchmark name is passed through to metrics."""
        port = _random_returns(50, seed=1)
        bench = _random_returns(50, seed=2)
        analyzer = BenchmarkAnalyzer()
        metrics = analyzer.calculate_benchmark_metrics(port, bench, "My Benchmark")
        assert metrics.benchmark_name == "My Benchmark"


# =============================================================================
# String benchmark lookup
# =============================================================================


class TestStringBenchmarkLookup:
    """Test that string keys resolve to BenchmarkDefinitions."""

    def test_spy_lookup(self):
        assert STANDARD_BENCHMARKS["SPY"].symbol == "SPY"
        assert STANDARD_BENCHMARKS["SPY"].name == "S&P 500"

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _ = STANDARD_BENCHMARKS["DOES_NOT_EXIST"]
