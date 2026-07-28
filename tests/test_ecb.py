"""Tests for ECB exchange rate client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.data.alternative.ecb import ECBClient, _breaker
from quant_tools.errors import DataError, NetworkError


def _sample_csv_response(n: int = 10) -> str:
    """Build a sample ECB CSV response."""
    lines = ["KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_CONF,OBS_PRE_BREAK,OBS_COM"]
    for i in range(n):
        day = f"2025-01-{i + 1:02d}"
        rate = 1.08 + i * 0.001
        lines.append(
            f"D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,{day},{rate:.4f},A,F,,"
        )
    return "\n".join(lines)


class TestECBClient:
    """Tests for ECBClient."""

    def setup_method(self) -> None:
        _breaker.reset()

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_success(self, mock_client_cls) -> None:
        """Test successful exchange rate fetch."""
        mock_response = MagicMock()
        mock_response.text = _sample_csv_response(10)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        result = client.fetch_exchange_rates("USD")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "date" in result.columns
        assert "rate" in result.columns
        assert result["date"].is_monotonic_increasing

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_empty(self, mock_client_cls) -> None:
        """Test empty response."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        with pytest.raises(DataError, match="No ECB exchange rate data"):
            client.fetch_exchange_rates("USD")

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_bad_format(self, mock_client_cls) -> None:
        """Test unexpected CSV format."""
        mock_response = MagicMock()
        mock_response.text = "col1,col2,col3\na,b,c\n"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        with pytest.raises(DataError, match="Unexpected ECB response format"):
            client.fetch_exchange_rates("USD")

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_timeout(self, mock_client_cls) -> None:
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_exchange_rates("USD")

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_http_error(self, mock_client_cls) -> None:
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

        client = ECBClient()
        with pytest.raises(NetworkError, match="500"):
            client.fetch_exchange_rates("USD")

    def test_circuit_breaker_open(self) -> None:
        """Test that circuit breaker blocks calls when open."""
        for _ in range(5):
            _breaker.record_failure()

        client = ECBClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_exchange_rates("USD")

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_different_currency(self, mock_client_cls) -> None:
        """Test fetching non-USD currency."""
        mock_response = MagicMock()
        mock_response.text = _sample_csv_response(5)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        result = client.fetch_exchange_rates("GBP", start="2020-01-01")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

        # Verify URL contains GBP
        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "GBP" in url

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_custom_start(self, mock_client_cls) -> None:
        """Test custom start date parameter."""
        mock_response = MagicMock()
        mock_response.text = _sample_csv_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        client.fetch_exchange_rates("USD", start="2020-01-01")

        # Verify startPeriod param
        call_args = mock_client.get.call_args
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        assert params["startPeriod"] == "2020-01-01"

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_sorted_by_date(self, mock_client_cls) -> None:
        """Test that results are sorted by date."""
        # Create CSV with dates out of order
        lines = ["KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_CONF,OBS_PRE_BREAK,OBS_COM"]
        lines.append("D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2025-01-05,1.0850,A,F,,")
        lines.append("D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2025-01-01,1.0800,A,F,,")
        lines.append("D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2025-01-03,1.0830,A,F,,")

        mock_response = MagicMock()
        mock_response.text = "\n".join(lines)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        result = client.fetch_exchange_rates("USD")

        assert len(result) == 3
        assert result["date"].is_monotonic_increasing

    @patch("quant_tools.data.alternative.ecb.httpx.Client")
    def test_fetch_exchange_rates_currency_case_insensitive(self, mock_client_cls) -> None:
        """Test that currency parameter is uppercased."""
        mock_response = MagicMock()
        mock_response.text = _sample_csv_response(3)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = ECBClient()
        client.fetch_exchange_rates("usd")

        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "USD" in url
