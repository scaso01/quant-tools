"""Tests for Wikipedia pageviews client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.data.alternative.wikipedia import WikipediaPageviews, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_pageviews_response(n: int = 10) -> dict:
    """Build a sample Wikimedia pageviews API response."""
    items = []
    for i in range(n):
        items.append(
            {
                "project": "en.wikipedia",
                "article": "Apple_Inc.",
                "granularity": "daily",
                "timestamp": f"202501{i + 10:02d}00",
                "access": "all-access",
                "agent": "all-agents",
                "views": 5000 + i * 100,
            }
        )
    return {"items": items}


class TestWikipediaPageviews:
    """Tests for WikipediaPageviews."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_success(self, mock_client_cls) -> None:
        """Test successful pageview fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_pageviews_response(10)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        result = client.fetch_pageviews("Apple_Inc.")

        assert len(result) == 10
        assert "date" in result.columns
        assert "views" in result.columns
        assert all(v > 0 for v in result["views"])

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_empty(self, mock_client_cls) -> None:
        """Test empty pageview response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        with pytest.raises(DataError, match="No Wikipedia pageviews"):
            client.fetch_pageviews("Nonexistent_Article_XYZ")

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_timeout(self, mock_client_cls) -> None:
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_pageviews("Apple_Inc.")

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_http_error(self, mock_client_cls) -> None:
        """Test HTTP error handling (e.g. 404 for unknown article)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        with pytest.raises(NetworkError, match="404"):
            client.fetch_pageviews("Apple_Inc.")

    def test_circuit_breaker_open(self) -> None:
        """Test that circuit breaker blocks calls when open."""
        for _ in range(5):
            _breaker.record_failure()

        client = WikipediaPageviews()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_pageviews("Apple_Inc.")

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_custom_dates(self, mock_client_cls) -> None:
        """Test custom date range parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_pageviews_response(5)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        result = client.fetch_pageviews(
            "Bitcoin", start="2025-01-01", end="2025-01-31"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

        # Verify URL contains formatted dates
        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "20250101" in url
        assert "20250131" in url

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_custom_project(self, mock_client_cls) -> None:
        """Test fetching from a different Wikipedia project."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_pageviews_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        result = client.fetch_pageviews(
            "Bitcoin", project="de.wikipedia", start="2025-01-01", end="2025-01-31"
        )
        assert len(result) == 3

        # Verify URL contains the project
        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "de.wikipedia" in url

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_spaces_to_underscores(self, mock_client_cls) -> None:
        """Test that spaces in article names are converted to underscores."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_pageviews_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        result = client.fetch_pageviews(
            "Apple Inc.", start="2025-01-01", end="2025-01-31"
        )
        assert len(result) == 3

        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "Apple_Inc." in url

    @patch("quant_tools.data.alternative.wikipedia.httpx.Client")
    def test_fetch_pageviews_date_format(self, mock_client_cls) -> None:
        """Test that dates are properly formatted in output."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_pageviews_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WikipediaPageviews()
        result = client.fetch_pageviews("Test", start="2025-01-01", end="2025-01-31")

        # Dates should be in YYYY-MM-DD format
        for date_str in result["date"]:
            assert len(date_str) == 10
            assert date_str[4] == "-"
            assert date_str[7] == "-"
