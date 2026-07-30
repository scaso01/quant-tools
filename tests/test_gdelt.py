"""Tests for GDELT event database client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_tools.data.alternative.gdelt import GdeltClient, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_events_df(n: int = 5) -> pd.DataFrame:
    """Build a sample GDELT events DataFrame."""
    return pd.DataFrame(
        {
            "SQLDATE": [f"2025010{i + 1}" for i in range(n)],
            "SOURCEURL": [f"https://example.com/article{i}" for i in range(n)],
            "AvgTone": [float(i - 2) for i in range(n)],
            "ActionGeo_FullName": ["United States"] * n,
            "ActionGeo_CountryCode": ["US"] * n,
            "Actor1Name": ["APPLE"] * n,
        }
    )


class TestGdeltClient:
    """Tests for GdeltClient."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_success(self, mock_ensure) -> None:
        """Test successful event fetch with mocked data."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = _sample_events_df(5)

        result = client.fetch_events("APPLE")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "date" in result.columns
        assert "title" in result.columns
        assert "url" in result.columns
        assert "tone" in result.columns
        assert "source_country" in result.columns

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_empty_result(self, mock_ensure) -> None:
        """Test empty result handling."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = pd.DataFrame()

        with pytest.raises(DataError, match="No GDELT events found"):
            client.fetch_events("NONEXISTENT_QUERY_XYZ")

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_none_result(self, mock_ensure) -> None:
        """Test None result from GDELT."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = None

        with pytest.raises(DataError, match="No GDELT events found"):
            client.fetch_events("TEST")

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_no_matching_query(self, mock_ensure) -> None:
        """Test when events exist but none match the query."""
        df = _sample_events_df(3)
        df["ActionGeo_FullName"] = "France"
        df["SOURCEURL"] = "https://other.com/article"
        df["Actor1Name"] = "OTHER"

        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = df

        with pytest.raises(DataError, match="No GDELT events matching"):
            client.fetch_events("ZZZZNOTFOUND")

    def test_circuit_breaker_open(self) -> None:
        """Test that circuit breaker blocks calls when open."""
        for _ in range(5):
            _breaker.record_failure()

        client = GdeltClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_events("AAPL")

    def test_library_not_installed(self) -> None:
        """Test graceful error when the gdelt library is missing."""
        client = GdeltClient()
        client._gd = None  # force re-import

        with patch.dict("sys.modules", {"gdelt": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'gdelt'"),
            ):
                with pytest.raises(DataError, match="gdelt not installed"):
                    client._ensure_client()

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_network_error(self, mock_ensure) -> None:
        """Test network error handling."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.side_effect = ConnectionError("Connection refused")

        with pytest.raises(NetworkError, match="unreachable"):
            client.fetch_events("AAPL")

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_custom_dates(self, mock_ensure) -> None:
        """Test custom date range."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = _sample_events_df(3)

        result = client.fetch_events(
            "APPLE", start_date="2025-01-01", end_date="2025-01-31"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_sentiment_timeline_success(self, mock_ensure) -> None:
        """Test successful sentiment timeline fetch."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = _sample_events_df(10)

        result = client.fetch_sentiment_timeline("APPLE", days=7)
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "avg_tone" in result.columns
        assert "article_count" in result.columns

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_sentiment_timeline_empty(self, mock_ensure) -> None:
        """Test empty sentiment timeline."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = pd.DataFrame()

        with pytest.raises(DataError, match="No GDELT sentiment data"):
            client.fetch_sentiment_timeline("NONEXISTENT")

    @patch("quant_tools.data.alternative.gdelt.GdeltClient._ensure_client")
    def test_fetch_events_max_records(self, mock_ensure) -> None:
        """Test max_records parameter limits output."""
        client = GdeltClient()
        client._gd = MagicMock()
        client._gd.Search.return_value = _sample_events_df(20)

        result = client.fetch_events("APPLE", max_records=3)
        assert len(result) <= 3
