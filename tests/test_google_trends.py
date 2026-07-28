"""Tests for Google Trends client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_tools.data.alternative.google_trends import GoogleTrendsClient, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_interest_df(keyword: str = "AAPL", n: int = 52) -> pd.DataFrame:
    """Build a sample interest-over-time DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    return pd.DataFrame(
        {keyword: range(20, 20 + n), "isPartial": [False] * n},
        index=dates,
    )


class TestGoogleTrendsClient:
    """Tests for GoogleTrendsClient."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_success(self, mock_ensure) -> None:
        """Test successful interest fetch with mocked data."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.return_value = _sample_interest_df("AAPL")

        result = client.fetch_interest("AAPL")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "interest" in result.columns

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_empty(self, mock_ensure) -> None:
        """Test empty result handling."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.return_value = pd.DataFrame()

        with pytest.raises(DataError, match="No Google Trends data"):
            client.fetch_interest("NONEXISTENT_XYZ")

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_none_result(self, mock_ensure) -> None:
        """Test None result from pytrends."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.return_value = None

        with pytest.raises(DataError, match="No Google Trends data"):
            client.fetch_interest("TEST")

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_drops_is_partial(self, mock_ensure) -> None:
        """Test that isPartial column is dropped."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.return_value = _sample_interest_df("BTC")

        result = client.fetch_interest("BTC")
        assert "isPartial" not in result.columns

    def test_circuit_breaker_open_interest(self) -> None:
        """Test that circuit breaker blocks interest calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = GoogleTrendsClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_interest("AAPL")

    def test_circuit_breaker_open_related(self) -> None:
        """Test that circuit breaker blocks related queries calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = GoogleTrendsClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_related_queries("AAPL")

    def test_library_not_installed(self) -> None:
        """Test graceful error when pytrends missing."""
        client = GoogleTrendsClient()
        client._client = None

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'pytrends'"),
        ):
            with pytest.raises(DataError, match="pytrends not installed"):
                client._ensure_client()

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_network_error(self, mock_ensure) -> None:
        """Test network error handling."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.side_effect = ConnectionError(
            "Connection refused"
        )

        # build_payload succeeds, interest_over_time fails
        with pytest.raises(NetworkError, match="unreachable"):
            client.fetch_interest("AAPL")

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_related_queries_success(self, mock_ensure) -> None:
        """Test successful related queries fetch."""
        client = GoogleTrendsClient()
        client._client = MagicMock()

        top_df = pd.DataFrame({"query": ["apple stock", "aapl price"], "value": [100, 85]})
        rising_df = pd.DataFrame({"query": ["apple ai"], "value": [500]})

        client._client.related_queries.return_value = {
            "AAPL": {"top": top_df, "rising": rising_df}
        }

        result = client.fetch_related_queries("AAPL")
        assert isinstance(result, dict)
        assert "top" in result
        assert "rising" in result
        assert len(result["top"]) == 2
        assert len(result["rising"]) == 1

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_related_queries_empty(self, mock_ensure) -> None:
        """Test empty related queries."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.related_queries.return_value = {}

        with pytest.raises(DataError, match="No related queries"):
            client.fetch_related_queries("NONEXISTENT_XYZ")

    @patch("quant_tools.data.alternative.google_trends.GoogleTrendsClient._ensure_client")
    def test_fetch_interest_with_geo(self, mock_ensure) -> None:
        """Test interest fetch with geo parameter."""
        client = GoogleTrendsClient()
        client._client = MagicMock()
        client._client.interest_over_time.return_value = _sample_interest_df("AAPL")

        result = client.fetch_interest("AAPL", geo="US")
        assert isinstance(result, pd.DataFrame)
        # Verify build_payload was called with geo
        client._client.build_payload.assert_called_once_with(
            ["AAPL"], cat=0, timeframe="today 12-m", geo="US", gprop=""
        )
