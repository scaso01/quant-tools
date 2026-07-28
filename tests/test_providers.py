"""Tests for data providers (~50 tests).

All external APIs are mocked. No real network calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.errors import DataError, NetworkError
from quant_tools.models import AssetClass


# =============================================================================
# Helpers
# =============================================================================


def _sample_ohlcv_df(n: int = 50) -> pd.DataFrame:
    """Build a sample OHLCV DataFrame for mock returns."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-06-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
            "Adj Close": close,
        },
        index=dates,
    )


def _sample_ccxt_ohlcv(n: int = 50) -> list[list]:
    """Build sample ccxt OHLCV list of lists."""
    np.random.seed(42)
    base_ts = int(datetime(2023, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    price = 30000.0
    for i in range(n):
        ts = base_ts + i * 86400000
        price *= 1 + np.random.normal(0, 0.02)
        rows.append([ts, price * 0.999, price * 1.005, price * 0.995, price, 1000.0 + i])
    return rows


# =============================================================================
# EquityProvider Tests
# =============================================================================


class TestEquityProvider:
    """Tests for EquityProvider."""

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_success(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = EquityProvider()
        df = provider.fetch_ohlcv("AAPL", period="1y")

        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        mock_yf.download.assert_called_once()

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_empty_raises_data_error(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = pd.DataFrame()

        provider = EquityProvider()
        with pytest.raises(DataError, match="No equity data"):
            provider.fetch_ohlcv("FAKE")

    @patch("quant_tools.data.providers.equity.yfinance_breaker")
    def test_fetch_ohlcv_circuit_breaker_open(self, mock_breaker):
        from quant_tools.data.providers.equity import EquityProvider

        mock_breaker.is_open = True

        provider = EquityProvider()
        with pytest.raises(NetworkError, match="circuit breaker"):
            provider.fetch_ohlcv("AAPL")

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_yfinance_exception_triggers_fallback(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.side_effect = Exception("Connection error")

        provider = EquityProvider()
        # Stooq fallback also fails -> DataError
        with patch.object(provider, "_stooq_fallback", return_value=None):
            with pytest.raises(DataError, match="No equity data"):
                provider.fetch_ohlcv("AAPL")

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_stooq_fallback_success(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.side_effect = Exception("yfinance down")

        provider = EquityProvider()
        fallback_df = _sample_ohlcv_df()
        with patch.object(provider, "_stooq_fallback", return_value=fallback_df):
            df = provider.fetch_ohlcv("AAPL")
            assert not df.empty

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_returns_expected_columns(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = EquityProvider()
        df = provider.fetch_ohlcv("MSFT")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_handles_multiindex_columns(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        raw = _sample_ohlcv_df()
        # Create MultiIndex columns like yfinance >= 0.2.40
        raw.columns = pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in raw.columns]
        )
        mock_yf.download.return_value = raw

        provider = EquityProvider()
        df = provider.fetch_ohlcv("AAPL")
        assert "Close" in df.columns

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_info_success(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider

        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,
        }
        mock_yf.Ticker.return_value = mock_ticker

        provider = EquityProvider()
        info = provider.fetch_info("AAPL")
        assert info["name"] == "Apple Inc."
        assert info["sector"] == "Technology"
        assert info["asset_class"] == "equity"

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_info_empty_raises_data_error(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider

        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        provider = EquityProvider()
        with pytest.raises(DataError, match="No info"):
            provider.fetch_info("FAKE")

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_info_exception_raises_data_error(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider

        mock_yf.Ticker.side_effect = Exception("API down")

        provider = EquityProvider()
        with pytest.raises(DataError, match="No info returned"):
            provider.fetch_info("AAPL")

    def test_name_is_equity(self):
        from quant_tools.data.providers.equity import EquityProvider

        provider = EquityProvider()
        assert provider.name == "equity"

    def test_supported_asset_classes(self):
        from quant_tools.data.providers.equity import EquityProvider

        provider = EquityProvider()
        assert AssetClass.EQUITY in provider.supported_asset_classes()

    @patch("quant_tools.data.providers.equity.yf")
    def test_fetch_ohlcv_records_breaker_success(self, mock_yf):
        from quant_tools.data.providers.equity import EquityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = EquityProvider()
        provider.fetch_ohlcv("AAPL")
        assert not yfinance_breaker.is_open


# =============================================================================
# CryptoProvider Tests
# =============================================================================


class TestCryptoProvider:
    """Tests for CryptoProvider."""

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_success(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider(exchange_id="binance")
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _sample_ccxt_ohlcv()
        provider._exchange = mock_exchange

        df = provider.fetch_ohlcv("BTC/USDT", period="1mo")
        assert not df.empty
        assert "Close" in df.columns
        mock_exchange.fetch_ohlcv.assert_called_once()

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_empty_raises_data_error(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider()
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        provider._exchange = mock_exchange

        with pytest.raises(DataError, match="No crypto data"):
            provider.fetch_ohlcv("BTC/USDT")

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_circuit_breaker_open(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = True

        provider = CryptoProvider()
        with pytest.raises(NetworkError, match="circuit breaker"):
            provider.fetch_ohlcv("BTC/USDT")

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_exception_raises_data_error(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider()
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Exchange down")
        provider._exchange = mock_exchange

        with pytest.raises(DataError, match="ccxt fetch failed"):
            provider.fetch_ohlcv("BTC/USDT")

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_returns_expected_columns(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider()
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _sample_ccxt_ohlcv()
        provider._exchange = mock_exchange

        df = provider.fetch_ohlcv("ETH/USDT")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_ohlcv_period_mapping(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider()
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _sample_ccxt_ohlcv(10)
        provider._exchange = mock_exchange

        df = provider.fetch_ohlcv("BTC/USDT", period="3mo")
        assert not df.empty

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_info_success(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider(exchange_id="binance")
        mock_exchange = MagicMock()
        mock_exchange.markets = {
            "BTC/USDT": {"base": "BTC", "quote": "USDT", "active": True},
        }
        provider._exchange = mock_exchange

        info = provider.fetch_info("BTC/USDT")
        assert info["base"] == "BTC"
        assert info["quote"] == "USDT"
        assert info["asset_class"] == "crypto"

    @patch("quant_tools.data.providers.crypto.ccxt_breaker")
    def test_fetch_info_missing_symbol(self, mock_breaker):
        from quant_tools.data.providers.crypto import CryptoProvider

        mock_breaker.is_open = False

        provider = CryptoProvider()
        mock_exchange = MagicMock()
        mock_exchange.markets = {}
        provider._exchange = mock_exchange

        with pytest.raises(DataError, match="not found"):
            provider.fetch_info("FAKE/USDT")

    def test_name_is_crypto(self):
        from quant_tools.data.providers.crypto import CryptoProvider

        provider = CryptoProvider()
        assert provider.name == "crypto"

    def test_supported_asset_classes(self):
        from quant_tools.data.providers.crypto import CryptoProvider

        provider = CryptoProvider()
        assert AssetClass.CRYPTO in provider.supported_asset_classes()

    def test_default_exchange_from_config(self):
        from quant_tools.data.providers.crypto import CryptoProvider

        provider = CryptoProvider()
        assert provider._exchange_id == "kraken"

    def test_custom_exchange(self):
        from quant_tools.data.providers.crypto import CryptoProvider

        provider = CryptoProvider(exchange_id="kraken")
        assert provider._exchange_id == "kraken"


# =============================================================================
# ForexProvider Tests
# =============================================================================


class TestForexProvider:
    """Tests for ForexProvider."""

    @patch("quant_tools.data.providers.forex.yf")
    def test_fetch_ohlcv_success(self, mock_yf):
        from quant_tools.data.providers.forex import ForexProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = ForexProvider()
        df = provider.fetch_ohlcv("EUR/USD")
        assert not df.empty
        mock_yf.download.assert_called_once()

    @patch("quant_tools.data.providers.forex.yf")
    def test_fetch_ohlcv_symbol_conversion(self, mock_yf):
        from quant_tools.data.providers.forex import ForexProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = ForexProvider()
        provider.fetch_ohlcv("EUR/USD")
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == "EURUSD=X"

    @patch("quant_tools.data.providers.forex.yf")
    def test_fetch_ohlcv_empty_raises_data_error(self, mock_yf):
        from quant_tools.data.providers.forex import ForexProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = pd.DataFrame()

        provider = ForexProvider()
        with pytest.raises(DataError, match="No forex data"):
            provider.fetch_ohlcv("EUR/USD")

    @patch("quant_tools.data.providers.forex.yfinance_breaker")
    def test_fetch_ohlcv_circuit_breaker_open(self, mock_breaker):
        from quant_tools.data.providers.forex import ForexProvider

        mock_breaker.is_open = True

        provider = ForexProvider()
        with pytest.raises(NetworkError, match="circuit breaker"):
            provider.fetch_ohlcv("EUR/USD")

    @patch("quant_tools.data.providers.forex.yf")
    def test_fetch_ohlcv_adds_volume_if_missing(self, mock_yf):
        from quant_tools.data.providers.forex import ForexProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        # Create df without Volume column
        df = _sample_ohlcv_df()
        df = df.drop(columns=["Volume"])
        mock_yf.download.return_value = df

        provider = ForexProvider()
        result = provider.fetch_ohlcv("EUR/USD")
        assert "Volume" in result.columns

    def test_fetch_info_basic(self):
        from quant_tools.data.providers.forex import ForexProvider

        provider = ForexProvider()
        info = provider.fetch_info("EUR/USD")
        assert info["base_currency"] == "EUR"
        assert info["quote_currency"] == "USD"
        assert info["asset_class"] == "forex"

    def test_fetch_info_no_slash(self):
        from quant_tools.data.providers.forex import ForexProvider

        provider = ForexProvider()
        info = provider.fetch_info("EURUSD")
        assert info["base_currency"] == "EUR"
        assert info["quote_currency"] == "USD"

    def test_name_is_forex(self):
        from quant_tools.data.providers.forex import ForexProvider

        provider = ForexProvider()
        assert provider.name == "forex"

    def test_supported_asset_classes(self):
        from quant_tools.data.providers.forex import ForexProvider

        provider = ForexProvider()
        assert AssetClass.FOREX in provider.supported_asset_classes()

    @patch("quant_tools.data.providers.forex.yf")
    def test_fetch_ohlcv_returns_expected_columns(self, mock_yf):
        from quant_tools.data.providers.forex import ForexProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = ForexProvider()
        df = provider.fetch_ohlcv("GBP/USD")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


# =============================================================================
# CommodityProvider Tests
# =============================================================================


class TestCommodityProvider:
    """Tests for CommodityProvider."""

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_ohlcv_success(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = CommodityProvider()
        df = provider.fetch_ohlcv("GOLD")
        assert not df.empty

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_ohlcv_maps_to_futures(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = CommodityProvider()
        provider.fetch_ohlcv("GOLD")
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == "GC=F"

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_ohlcv_oil_maps_to_futures(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = CommodityProvider()
        provider.fetch_ohlcv("OIL")
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == "CL=F"

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_ohlcv_empty_raises_data_error(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = pd.DataFrame()

        provider = CommodityProvider()
        with pytest.raises(DataError, match="No commodity data"):
            provider.fetch_ohlcv("GOLD")

    @patch("quant_tools.data.providers.commodity.yfinance_breaker")
    def test_fetch_ohlcv_circuit_breaker_open(self, mock_breaker):
        from quant_tools.data.providers.commodity import CommodityProvider

        mock_breaker.is_open = True

        provider = CommodityProvider()
        with pytest.raises(NetworkError, match="circuit breaker"):
            provider.fetch_ohlcv("GOLD")

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_info_success(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider

        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "SPDR Gold Shares"}
        mock_yf.Ticker.return_value = mock_ticker

        provider = CommodityProvider()
        info = provider.fetch_info("GOLD")
        assert info["ticker"] == "GC=F"
        assert info["asset_class"] == "commodity"

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_info_unknown_commodity(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider

        # Direct ticker (short alpha string) should pass through
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        provider = CommodityProvider()
        info = provider.fetch_info("SLV")
        assert info["ticker"] == "SLV"

    def test_resolve_symbol_known(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        assert provider._resolve_symbol("GOLD") == "GC=F"
        assert provider._resolve_symbol("OIL") == "CL=F"
        assert provider._resolve_symbol("SILVER") == "SI=F"

    def test_resolve_symbol_futures_passthrough(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        assert provider._resolve_symbol("GC=F") == "GC=F"

    def test_resolve_symbol_direct_ticker(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        assert provider._resolve_symbol("GLD") == "GLD"

    def test_resolve_symbol_unknown_raises(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        with pytest.raises(DataError, match="No commodity symbol"):
            provider._resolve_symbol("UNKNOWN_LONG_NAME")

    def test_name_is_commodity(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        assert provider.name == "commodity"

    def test_supported_asset_classes(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = CommodityProvider()
        assert AssetClass.COMMODITY in provider.supported_asset_classes()

    @patch("quant_tools.data.providers.commodity.yf")
    def test_fetch_ohlcv_returns_expected_columns(self, mock_yf):
        from quant_tools.data.providers.commodity import CommodityProvider
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = _sample_ohlcv_df()

        provider = CommodityProvider()
        df = provider.fetch_ohlcv("SILVER")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
