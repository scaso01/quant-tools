"""Tests for CoinGecko alternative data client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from quant_tools.data.alternative.coingecko import CoinGeckoClient
from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import coingecko_breaker


def _sample_market_chart(n: int = 10) -> dict:
    """Build a sample CoinGecko market_chart response."""
    base_ts = 1700000000000  # milliseconds
    return {
        "prices": [
            [base_ts + i * 86400000, 40000.0 + i * 100]
            for i in range(n)
        ],
        "market_caps": [
            [base_ts + i * 86400000, 800_000_000_000.0 + i * 1e9]
            for i in range(n)
        ],
        "total_volumes": [
            [base_ts + i * 86400000, 20_000_000_000.0 + i * 1e8]
            for i in range(n)
        ],
    }


def _sample_trending() -> dict:
    """Build a sample trending response."""
    return {
        "coins": [
            {
                "item": {
                    "id": f"coin{i}",
                    "name": f"Coin {i}",
                    "symbol": f"C{i}",
                    "market_cap_rank": i + 1,
                    "score": i,
                }
            }
            for i in range(7)
        ]
    }


def _sample_categories() -> list[dict]:
    """Build a sample categories response."""
    return [
        {
            "id": f"cat-{i}",
            "name": f"Category {i}",
            "market_cap": 1e12 - i * 1e11,
            "total_volume": 5e10 - i * 1e9,
            "market_cap_change_24h": -2.0 + i * 0.5,
        }
        for i in range(5)
    ]


class TestCoinGeckoClient:
    """Tests for CoinGeckoClient."""

    def setup_method(self) -> None:
        coingecko_breaker.reset()

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_market_data_success(self, mock_client_cls) -> None:
        """Test successful market data fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_market_chart(10)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        result = client.fetch_market_data("bitcoin", days=10)

        assert len(result) > 0
        assert "date" in result.columns
        assert "price" in result.columns
        assert "market_cap" in result.columns
        assert "volume" in result.columns

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_market_data_empty(self, mock_client_cls) -> None:
        """Test empty market data response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"prices": [], "market_caps": [], "total_volumes": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        with pytest.raises(DataError, match="No CoinGecko market data"):
            client.fetch_market_data("nonexistent_coin_xyz")

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_market_data_timeout(self, mock_client_cls) -> None:
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_market_data("bitcoin")

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_market_data_http_error(self, mock_client_cls) -> None:
        """Test HTTP error (e.g. rate limited)."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        with pytest.raises(NetworkError, match="429"):
            client.fetch_market_data("bitcoin")

    def test_circuit_breaker_open_market(self) -> None:
        """Test that circuit breaker blocks market data calls."""
        for _ in range(5):
            coingecko_breaker.record_failure()

        client = CoinGeckoClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_market_data("bitcoin")

    def test_circuit_breaker_open_trending(self) -> None:
        """Test that circuit breaker blocks trending calls."""
        for _ in range(5):
            coingecko_breaker.record_failure()

        client = CoinGeckoClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_trending()

    def test_circuit_breaker_open_categories(self) -> None:
        """Test that circuit breaker blocks categories calls."""
        for _ in range(5):
            coingecko_breaker.record_failure()

        client = CoinGeckoClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_categories()

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_trending_success(self, mock_client_cls) -> None:
        """Test successful trending fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_trending()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        result = client.fetch_trending()

        assert isinstance(result, list)
        assert len(result) == 7
        assert "id" in result[0]
        assert "name" in result[0]
        assert "symbol" in result[0]

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_trending_empty(self, mock_client_cls) -> None:
        """Test empty trending response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"coins": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        with pytest.raises(DataError, match="No trending coins"):
            client.fetch_trending()

    @patch("quant_tools.data.alternative.coingecko.httpx.Client")
    def test_fetch_categories_success(self, mock_client_cls) -> None:
        """Test successful categories fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_categories()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = CoinGeckoClient()
        result = client.fetch_categories()

        assert isinstance(result, list)
        assert len(result) == 5
        assert "id" in result[0]
        assert "name" in result[0]
        assert "market_cap" in result[0]
        assert "volume_24h" in result[0]
