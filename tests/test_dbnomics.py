"""Tests for DBnomics client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.data.alternative.dbnomics import DBnomicsClient, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_series_response(n: int = 10) -> dict:
    """Build a sample DBnomics series API response."""
    return {
        "series": {
            "docs": [
                {
                    "period": [f"2020-{i + 1:02d}" for i in range(n)],
                    "value": [100.0 + i * 5 for i in range(n)],
                }
            ]
        }
    }


def _sample_search_response(n: int = 3) -> dict:
    """Build a sample DBnomics search API response."""
    return {
        "results": [
            {
                "provider_code": f"PROV{i}",
                "dataset_code": f"DS{i}",
                "series_code": f"SER{i}",
                "series_name": f"Series {i}",
                "dataset_name": f"Dataset {i}",
            }
            for i in range(n)
        ]
    }


class TestDBnomicsClient:
    """Tests for DBnomicsClient."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_success(self, mock_client_cls) -> None:
        """Test successful series fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_series_response(10)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        result = client.fetch_series("IMF", "WEO", "USA.NGDP_RPCH")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "value" in result.columns
        assert result.index.name == "date"

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_empty_docs(self, mock_client_cls) -> None:
        """Test empty series response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"series": {"docs": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(DataError, match="No DBnomics series found"):
            client.fetch_series("IMF", "WEO", "NONEXISTENT")

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_empty_periods(self, mock_client_cls) -> None:
        """Test series with empty periods."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "series": {"docs": [{"period": [], "value": []}]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(DataError, match="Empty DBnomics series"):
            client.fetch_series("IMF", "WEO", "EMPTY")

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_all_na_values(self, mock_client_cls) -> None:
        """Test series where all values are NA."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "series": {"docs": [{"period": ["2020-01", "2020-02"], "value": ["NA", "NA"]}]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(DataError, match="All values null"):
            client.fetch_series("IMF", "WEO", "ALLNA")

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_timeout(self, mock_client_cls) -> None:
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_series("IMF", "WEO", "TEST")

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_fetch_series_http_error(self, mock_client_cls) -> None:
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(NetworkError, match="500"):
            client.fetch_series("IMF", "WEO", "TEST")

    def test_circuit_breaker_open_series(self) -> None:
        """Test that circuit breaker blocks series calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = DBnomicsClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_series("IMF", "WEO", "TEST")

    def test_circuit_breaker_open_search(self) -> None:
        """Test that circuit breaker blocks search calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = DBnomicsClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.search("GDP")

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_search_success(self, mock_client_cls) -> None:
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_search_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        result = client.search("US GDP", limit=3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "provider_code" in result[0]
        assert "dataset_code" in result[0]
        assert "name" in result[0]

    @patch("quant_tools.data.alternative.dbnomics.httpx.Client")
    def test_search_empty(self, mock_client_cls) -> None:
        """Test empty search results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = DBnomicsClient()
        with pytest.raises(DataError, match="No DBnomics results"):
            client.search("ZZZZNONEXISTENT")
