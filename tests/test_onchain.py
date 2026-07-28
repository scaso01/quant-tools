"""Tests for DefiLlamaClient (on-chain data provider).

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.data.providers.onchain import DefiLlamaClient
from quant_tools.errors import DataError, NetworkError


def _mock_httpx_response(json_data: dict | list, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


class TestFetchTvl:
    """Tests for DefiLlamaClient.fetch_tvl."""

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_tvl_success(self, mock_breaker):
        """Mock httpx and verify DataFrame structure."""
        mock_breaker.is_open = False

        tvl_data = {
            "tvl": [
                {"date": 1672531200, "totalLiquidityUSD": 5_000_000_000},
                {"date": 1672617600, "totalLiquidityUSD": 5_100_000_000},
                {"date": 1672704000, "totalLiquidityUSD": 4_900_000_000},
            ]
        }

        mock_resp = _mock_httpx_response(tvl_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_tvl("aave")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "tvl"]
        assert len(df) == 3
        assert df["tvl"].iloc[0] == 5_000_000_000
        mock_breaker.record_success.assert_called_once()

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_tvl_unknown_protocol(self, mock_breaker):
        """Empty TVL data raises DataError."""
        mock_breaker.is_open = False

        mock_resp = _mock_httpx_response({"tvl": []})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            with pytest.raises(DataError, match="No TVL data"):
                client.fetch_tvl("nonexistent-protocol")

        mock_breaker.record_failure.assert_called_once()

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_circuit_breaker_open(self, mock_breaker):
        """NetworkError when breaker is open."""
        mock_breaker.is_open = True

        client = DefiLlamaClient()
        with pytest.raises(NetworkError, match="circuit breaker is open"):
            client.fetch_tvl("aave")

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_network_timeout(self, mock_breaker):
        """httpx timeout raises NetworkError."""
        mock_breaker.is_open = False

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            with pytest.raises(NetworkError, match="timeout"):
                client.fetch_tvl("aave")

        mock_breaker.record_failure.assert_called_once()


class TestFetchYields:
    """Tests for DefiLlamaClient.fetch_yields."""

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_yields_all_chains(self, mock_breaker):
        """No chain filter returns all pools above min_tvl."""
        mock_breaker.is_open = False

        pools_data = {
            "data": [
                {"symbol": "USDC-USDT", "chain": "Ethereum", "tvlUsd": 5_000_000, "apy": 3.5, "project": "aave"},
                {"symbol": "ETH-USDC", "chain": "Arbitrum", "tvlUsd": 2_000_000, "apy": 8.2, "project": "uniswap"},
                {"symbol": "TINY", "chain": "Ethereum", "tvlUsd": 500, "apy": 50.0, "project": "tiny"},  # Below min_tvl
            ]
        }

        mock_resp = _mock_httpx_response(pools_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_yields(min_tvl=1e6)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # TINY excluded
        assert list(df.columns) == ["pool", "chain", "tvl", "apy", "project"]
        # Sorted by APY descending
        assert df.iloc[0]["apy"] == 8.2

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_yields_filtered(self, mock_breaker):
        """Filter to specific chain."""
        mock_breaker.is_open = False

        pools_data = {
            "data": [
                {"symbol": "USDC-USDT", "chain": "Ethereum", "tvlUsd": 5_000_000, "apy": 3.5, "project": "aave"},
                {"symbol": "ETH-USDC", "chain": "Arbitrum", "tvlUsd": 2_000_000, "apy": 8.2, "project": "uniswap"},
            ]
        }

        mock_resp = _mock_httpx_response(pools_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_yields(chain="Ethereum", min_tvl=1e6)

        assert len(df) == 1
        assert df.iloc[0]["chain"] == "Ethereum"

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_min_tvl_filter(self, mock_breaker):
        """Yields below min_tvl threshold are excluded."""
        mock_breaker.is_open = False

        pools_data = {
            "data": [
                {"symbol": "BIG", "chain": "Ethereum", "tvlUsd": 10_000_000, "apy": 5.0, "project": "aave"},
                {"symbol": "SMALL", "chain": "Ethereum", "tvlUsd": 100_000, "apy": 20.0, "project": "small"},
            ]
        }

        mock_resp = _mock_httpx_response(pools_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_yields(min_tvl=5e6)

        assert len(df) == 1
        assert df.iloc[0]["pool"] == "BIG"


class TestFetchStablecoins:
    """Tests for DefiLlamaClient.fetch_stablecoins."""

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_stablecoins(self, mock_breaker):
        """Verify columns and structure."""
        mock_breaker.is_open = False

        stablecoin_data = {
            "peggedAssets": [
                {
                    "name": "Tether",
                    "symbol": "USDT",
                    "pegType": "peggedUSD",
                    "circulating": {"peggedUSD": 83_000_000_000},
                    "chainCirculating": {"Ethereum": {}, "Tron": {}, "BSC": {}, "Solana": {}},
                },
                {
                    "name": "USD Coin",
                    "symbol": "USDC",
                    "pegType": "peggedUSD",
                    "circulating": {"peggedUSD": 32_000_000_000},
                    "chainCirculating": {"Ethereum": {}, "Solana": {}},
                },
            ]
        }

        mock_resp = _mock_httpx_response(stablecoin_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_stablecoins()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "symbol", "peg", "mcap", "chain"]
        assert len(df) == 2
        # Sorted by mcap descending
        assert df.iloc[0]["symbol"] == "USDT"
        assert df.iloc[0]["mcap"] == 83_000_000_000
        mock_breaker.record_success.assert_called_once()


class TestFetchFees:
    """Tests for DefiLlamaClient.fetch_fees."""

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_fetch_fees_success(self, mock_breaker):
        """Mock response and verify DataFrame."""
        mock_breaker.is_open = False

        fees_data = {
            "totalDataChart": [
                [1672531200, 500_000],
                [1672617600, 600_000],
            ],
            "totalDataChartBreakdown": [
                [1672531200, {"uniswap": {"Revenue": 100_000}}],
                [1672617600, {"uniswap": {"Revenue": 120_000}}],
            ],
        }

        mock_resp = _mock_httpx_response(fees_data)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            df = client.fetch_fees("uniswap")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "fees", "revenue"]
        assert len(df) == 2
        assert df["fees"].iloc[0] == 500_000
        assert df["revenue"].iloc[0] == 100_000
        mock_breaker.record_success.assert_called_once()

    @patch("quant_tools.data.providers.onchain.defillama_breaker")
    def test_empty_response(self, mock_breaker):
        """Empty fee data raises DataError."""
        mock_breaker.is_open = False

        mock_resp = _mock_httpx_response({"totalDataChart": []})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            client = DefiLlamaClient()
            with pytest.raises(DataError, match="No fee data"):
                client.fetch_fees("nonexistent")

        mock_breaker.record_failure.assert_called_once()
