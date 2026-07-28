"""Tests for cross-asset correlation analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, CrossAssetCorrelation
from quant_tools.portfolio.cross_correlation import (
    calculate_cross_asset_correlations,
    calculate_rolling_correlations,
)


# =============================================================================
# Test calculate_rolling_correlations
# =============================================================================


class TestRollingCorrelations:
    """Test rolling pairwise correlation calculation."""

    def test_perfect_positive_correlation(self):
        """Identical series should yield rolling correlation of 1.0."""
        np.random.seed(42)
        a = np.random.normal(0, 0.01, 200)
        correlations, indices = calculate_rolling_correlations(a, a.copy(), window=63)
        assert len(correlations) > 0
        for c in correlations:
            assert c == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Negated series should yield rolling correlation of -1.0."""
        np.random.seed(42)
        a = np.random.normal(0, 0.01, 200)
        correlations, indices = calculate_rolling_correlations(a, -a, window=63)
        assert len(correlations) > 0
        for c in correlations:
            assert c == pytest.approx(-1.0)

    def test_uncorrelated_series(self):
        """Independent random series should have correlation near 0."""
        np.random.seed(42)
        a = np.random.normal(0, 0.01, 500)
        np.random.seed(99)
        b = np.random.normal(0, 0.01, 500)
        correlations, indices = calculate_rolling_correlations(a, b, window=63)
        assert len(correlations) > 0
        # Average of rolling correlations should be near zero
        avg = np.mean(correlations)
        assert abs(avg) < 0.2

    def test_correct_number_of_windows(self):
        """Rolling correlation should return n - window + 1 values."""
        np.random.seed(42)
        n = 100
        a = np.random.normal(0, 0.01, n)
        b = np.random.normal(0, 0.01, n)
        window = 20
        correlations, indices = calculate_rolling_correlations(a, b, window=window)
        assert len(correlations) == n - window + 1
        assert len(indices) == n - window + 1

    def test_window_size_respected(self):
        """Indices should start at window-1 and end at n-1."""
        np.random.seed(42)
        n = 100
        a = np.random.normal(0, 0.01, n)
        b = np.random.normal(0, 0.01, n)
        window = 30
        correlations, indices = calculate_rolling_correlations(a, b, window=window)
        assert indices[0] == window - 1
        assert indices[-1] == n - 1

    def test_short_series_returns_empty(self):
        """Series shorter than window should return empty lists."""
        a = np.array([0.01, -0.02, 0.03])
        b = np.array([0.02, -0.01, 0.04])
        correlations, indices = calculate_rolling_correlations(a, b, window=63)
        assert correlations == []
        assert indices == []

    def test_nan_handling(self):
        """Constant windows (zero std) produce NaN internally, returned as 0.0."""
        # Create a series that has a constant window
        a = np.zeros(100)
        b = np.random.normal(0, 0.01, 100)
        correlations, indices = calculate_rolling_correlations(a, b, window=20)
        # All correlations with a constant series should be 0.0 (NaN -> 0.0)
        for c in correlations:
            assert c == 0.0

    def test_different_length_arrays(self):
        """Arrays of different length should use the shorter one."""
        np.random.seed(42)
        a = np.random.normal(0, 0.01, 200)
        b = np.random.normal(0, 0.01, 150)
        correlations, indices = calculate_rolling_correlations(a, b, window=63)
        # Should use n=150 (the shorter)
        assert len(correlations) == 150 - 63 + 1


# =============================================================================
# Test CrossAssetCorrelation matrix properties
# =============================================================================


class TestCrossAssetCorrelationMatrix:
    """Test correlation matrix properties."""

    def test_matrix_is_square(self):
        """Correlation matrix should be square."""
        np.random.seed(42)
        n = 100
        returns_df = pd.DataFrame({
            "A": np.random.normal(0, 0.01, n),
            "B": np.random.normal(0, 0.02, n),
            "C": np.random.normal(0, 0.015, n),
        })
        corr_matrix = returns_df.corr().values
        result = CrossAssetCorrelation(
            symbols=["A", "B", "C"],
            asset_classes=[AssetClass.EQUITY, AssetClass.CRYPTO, AssetClass.FOREX],
            correlation_matrix=corr_matrix,
        )
        assert result.correlation_matrix.shape[0] == result.correlation_matrix.shape[1]
        assert result.correlation_matrix.shape[0] == 3

    def test_matrix_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        np.random.seed(42)
        n = 100
        returns_df = pd.DataFrame({
            "A": np.random.normal(0, 0.01, n),
            "B": np.random.normal(0, 0.02, n),
        })
        corr_matrix = returns_df.corr().values
        for i in range(corr_matrix.shape[0]):
            assert corr_matrix[i, i] == pytest.approx(1.0)


# =============================================================================
# Test calculate_cross_asset_correlations (mocked providers)
# =============================================================================


class TestCalculateCrossAssetCorrelations:
    """Test cross-asset correlation with mocked providers."""

    def test_with_mocked_providers(self):
        """Full cross-asset correlation with mocked provider data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="B")

        # Create synthetic OHLCV data for two symbols
        def make_ohlcv(seed: int) -> pd.DataFrame:
            np.random.seed(seed)
            close = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            return pd.DataFrame(
                {
                    "Open": close * 0.99,
                    "High": close * 1.01,
                    "Low": close * 0.98,
                    "Close": close,
                    "Volume": np.random.randint(1000, 10000, len(dates)),
                },
                index=dates,
            )

        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.side_effect = [make_ohlcv(42), make_ohlcv(99)]

        mock_registry = MagicMock()
        mock_registry.detect_asset_class.return_value = AssetClass.EQUITY
        mock_registry.get_provider.return_value = mock_provider

        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            result = calculate_cross_asset_correlations(
                symbols=["AAPL", "MSFT"],
                period="1y",
            )

        assert len(result.symbols) == 2
        assert result.correlation_matrix is not None
        assert result.correlation_matrix.shape == (2, 2)
        # Diagonal should be 1.0
        assert result.correlation_matrix[0, 0] == pytest.approx(1.0)
        assert result.correlation_matrix[1, 1] == pytest.approx(1.0)

    def test_single_symbol_returns_no_matrix(self):
        """With only one symbol, correlation matrix should be None."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        close = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        ohlcv = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": np.random.randint(1000, 10000, len(dates)),
            },
            index=dates,
        )

        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.return_value = ohlcv

        mock_registry = MagicMock()
        mock_registry.detect_asset_class.return_value = AssetClass.EQUITY
        mock_registry.get_provider.return_value = mock_provider

        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            result = calculate_cross_asset_correlations(
                symbols=["AAPL"],
                period="1y",
            )

        assert result.correlation_matrix is None

    def test_provider_failure_handled(self):
        """If a provider raises, that symbol is skipped gracefully."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        close = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        ohlcv = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": np.random.randint(1000, 10000, len(dates)),
            },
            index=dates,
        )

        mock_provider = MagicMock()
        # First call succeeds, second raises
        mock_provider.fetch_ohlcv.side_effect = [ohlcv, RuntimeError("API down")]

        mock_registry = MagicMock()
        mock_registry.detect_asset_class.return_value = AssetClass.EQUITY
        mock_registry.get_provider.return_value = mock_provider

        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            result = calculate_cross_asset_correlations(
                symbols=["AAPL", "FAIL"],
                period="1y",
            )

        # Only one symbol succeeded, so no correlation matrix
        assert result.correlation_matrix is None

    def test_explicit_asset_classes(self):
        """Passing explicit asset_classes skips auto-detection."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="B")

        def make_ohlcv(seed: int) -> pd.DataFrame:
            np.random.seed(seed)
            close = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            return pd.DataFrame(
                {
                    "Open": close * 0.99,
                    "High": close * 1.01,
                    "Low": close * 0.98,
                    "Close": close,
                    "Volume": np.random.randint(1000, 10000, len(dates)),
                },
                index=dates,
            )

        mock_provider = MagicMock()
        mock_provider.fetch_ohlcv.side_effect = [make_ohlcv(42), make_ohlcv(99)]

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            result = calculate_cross_asset_correlations(
                symbols=["AAPL", "BTC/USDT"],
                period="1y",
                asset_classes=[AssetClass.EQUITY, AssetClass.CRYPTO],
            )

        # detect_asset_class should NOT have been called
        mock_registry.detect_asset_class.assert_not_called()
        assert result.asset_classes == [AssetClass.EQUITY, AssetClass.CRYPTO]
        assert result.correlation_matrix is not None
