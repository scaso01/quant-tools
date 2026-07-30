"""Tests for Fear & Greed index client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

import httpx
import pytest

from quant_tools.data.alternative.fear_greed import FearGreedClient, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_crypto_response(n: int = 5) -> dict:
    """Build a sample crypto Fear & Greed API response."""
    data = []
    for i in range(n):
        value = 20 + i * 15
        if value <= 25:
            classification = "Extreme Fear"
        elif value <= 45:
            classification = "Fear"
        elif value <= 55:
            classification = "Neutral"
        elif value <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"

        data.append(
            {
                "value": str(value),
                "value_classification": classification,
                "timestamp": str(1700000000 + i * 86400),
            }
        )
    return {"data": data}


class TestFearGreedClient:
    """Tests for FearGreedClient."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.fear_greed.httpx.Client")
    def test_fetch_crypto_index_success(self, mock_client_cls) -> None:
        """Test successful crypto Fear & Greed fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = _sample_crypto_response(5)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = FearGreedClient()
        result = client.fetch_crypto_index(limit=5)

        assert len(result) == 5
        assert "date" in result.columns
        assert "value" in result.columns
        assert "classification" in result.columns
        assert all(0 <= v <= 100 for v in result["value"])

    @patch("quant_tools.data.alternative.fear_greed.httpx.Client")
    def test_fetch_crypto_index_empty(self, mock_client_cls) -> None:
        """Test empty crypto response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = FearGreedClient()
        with pytest.raises(DataError, match="No crypto Fear & Greed data"):
            client.fetch_crypto_index()

    @patch("quant_tools.data.alternative.fear_greed.httpx.Client")
    def test_fetch_crypto_index_no_data_key(self, mock_client_cls) -> None:
        """Test response missing 'data' key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = FearGreedClient()
        with pytest.raises(DataError, match="No crypto Fear & Greed data"):
            client.fetch_crypto_index()

    @patch("quant_tools.data.alternative.fear_greed.httpx.Client")
    def test_fetch_crypto_index_timeout(self, mock_client_cls) -> None:
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = FearGreedClient()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_crypto_index()

    @patch("quant_tools.data.alternative.fear_greed.httpx.Client")
    def test_fetch_crypto_index_http_error(self, mock_client_cls) -> None:
        """Test HTTP error handling."""
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

        client = FearGreedClient()
        with pytest.raises(NetworkError, match="429"):
            client.fetch_crypto_index()

    def test_circuit_breaker_open_crypto(self) -> None:
        """Test that circuit breaker blocks crypto calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = FearGreedClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_crypto_index()

    def test_circuit_breaker_open_cnn(self) -> None:
        """Test that circuit breaker blocks CNN calls."""
        for _ in range(5):
            _breaker.record_failure()

        client = FearGreedClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_cnn_index()

    def test_cnn_library_not_installed(self) -> None:
        """Test graceful error when fear-and-greed missing."""
        client = FearGreedClient()
        client._cnn_client = None

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'fear_and_greed'"),
        ):
            with pytest.raises(DataError, match="fear-and-greed not installed"):
                client._ensure_cnn_client()

    @patch("quant_tools.data.alternative.fear_greed.FearGreedClient._ensure_cnn_client")
    def test_fetch_cnn_index_success(self, mock_ensure) -> None:
        """Test successful CNN Fear & Greed fetch."""
        # A MagicMock answers to any attribute, so the old version of this test
        # invented a `previous_close` the real library has never returned and
        # passed while the command failed. fear-and-greed returns exactly this
        # namedtuple, so use it and let a shape change break the test.
        FearGreedIndex = namedtuple("FearGreedIndex", ["value", "description", "last_update"])
        mock_module = MagicMock()
        mock_module.get.return_value = FearGreedIndex(
            value=65.5, description="Greed", last_update="2025-01-15T12:00:00Z"
        )

        client = FearGreedClient()
        client._cnn_client = mock_module

        result = client.fetch_cnn_index()
        assert isinstance(result, dict)
        assert result["score"] == 65.5
        assert result["rating"] == "Greed"
        assert result["last_update"] == "2025-01-15T12:00:00Z"
        assert "previous_close" not in result

    @patch("quant_tools.data.alternative.fear_greed.FearGreedClient._ensure_cnn_client")
    def test_fetch_cnn_index_network_error(self, mock_ensure) -> None:
        """Test CNN network error handling."""
        mock_module = MagicMock()
        mock_module.get.side_effect = ConnectionError("Connection refused")

        client = FearGreedClient()
        client._cnn_client = mock_module

        with pytest.raises(NetworkError, match="unreachable"):
            client.fetch_cnn_index()
