"""Tests for World Bank API client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.errors import DataError, NetworkError


def _mock_worldbank_response(records: list[dict]) -> list:
    """Build a mock World Bank API response."""
    metadata = {"page": 1, "pages": 1, "per_page": 500, "total": len(records)}
    return [metadata, records]


def _sample_records(n: int = 5, start_year: int = 2020) -> list[dict]:
    """Build sample World Bank API records."""
    return [
        {
            "date": str(start_year + i),
            "value": 100.0 + i * 10,
            "country": {"value": "United States"},
            "indicator": {"value": "GDP (current US$)"},
        }
        for i in range(n)
    ]


class TestWorldBankClient:
    """Tests for WorldBankClient."""

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_success(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        records = _sample_records(5)
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_worldbank_response(records)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        df = client.fetch_indicator("US", "gdp")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "year" in df.columns
        assert "value" in df.columns
        assert "country" in df.columns
        assert df["year"].is_monotonic_increasing

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_shortname_resolved(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import INDICATORS, WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        records = _sample_records(3)
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_worldbank_response(records)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        client.fetch_indicator("US", "inflation")

        # Check the URL contains the resolved indicator code
        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert INDICATORS["inflation"] in url

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_no_data(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        mock_response = MagicMock()
        mock_response.json.return_value = [{"page": 1, "total": 0}, None]
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        with pytest.raises(DataError, match="No World Bank data"):
            client.fetch_indicator("XX", "gdp")

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_all_null_values(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        records = [
            {"date": "2020", "value": None, "country": {"value": "US"}, "indicator": {"value": "GDP"}},
            {"date": "2021", "value": None, "country": {"value": "US"}, "indicator": {"value": "GDP"}},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_worldbank_response(records)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        with pytest.raises(DataError, match="All values null"):
            client.fetch_indicator("US", "gdp")

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_timeout(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        with pytest.raises(NetworkError, match="timeout"):
            client.fetch_indicator("US", "gdp")

        worldbank_breaker.reset()

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_http_error(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        with pytest.raises(NetworkError, match="500"):
            client.fetch_indicator("US", "gdp")

        worldbank_breaker.reset()

    def test_circuit_breaker_open(self):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()
        for _ in range(5):
            worldbank_breaker.record_failure()

        client = WorldBankClient()
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_indicator("US", "gdp")

        worldbank_breaker.reset()

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_indicator_custom_date_range(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        records = _sample_records(3, start_year=2010)
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_worldbank_response(records)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        # The assertion is on the outgoing request, not the frame.
        client.fetch_indicator("CN", "gdp_growth", date_range="2010:2012")

        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["date"] == "2010:2012"

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_multiple_indicators(self, mock_client_cls):
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        records = _sample_records(3)
        mock_response = MagicMock()
        mock_response.json.return_value = _mock_worldbank_response(records)
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        df = client.fetch_multiple_indicators("US", indicators=["gdp", "inflation"])

        assert isinstance(df, pd.DataFrame)
        assert "year" in df.columns
        # Should have merged two indicator columns
        assert len(df.columns) >= 2

    @patch("quant_tools.data.providers.worldbank.httpx.Client")
    def test_fetch_multiple_indicators_partial_failure(self, mock_client_cls):
        """One indicator failing shouldn't break the whole fetch."""
        from quant_tools.data.providers.worldbank import WorldBankClient
        from quant_tools.retry import worldbank_breaker

        worldbank_breaker.reset()

        call_count = [0]

        def mock_get(url, params=None):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if call_count[0] == 1:
                mock_resp.json.return_value = _mock_worldbank_response(_sample_records(3))
            else:
                # Second call returns no data
                mock_resp.json.return_value = [{"page": 1, "total": 0}, None]
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = mock_get
        mock_client_cls.return_value = mock_client

        client = WorldBankClient()
        df = client.fetch_multiple_indicators("US", indicators=["gdp", "inflation"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Only first indicator's data

    def test_indicators_dict_has_expected_keys(self):
        from quant_tools.data.providers.worldbank import INDICATORS

        expected = {"gdp", "gdp_growth", "inflation", "unemployment", "trade_balance",
                    "interest_rate", "population", "gdp_per_capita"}
        assert set(INDICATORS.keys()) == expected
