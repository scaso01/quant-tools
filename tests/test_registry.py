"""Tests for ProviderRegistry (~25 tests).

All external APIs are mocked. No real network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.data.providers.registry import ProviderRegistry
from quant_tools.models import AssetClass


# =============================================================================
# detect_asset_class Tests
# =============================================================================


class TestDetectAssetClass:
    """Tests for ProviderRegistry.detect_asset_class."""

    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_crypto_with_slash(self):
        assert self.registry.detect_asset_class("BTC/USDT") == AssetClass.CRYPTO

    def test_crypto_eth_btc(self):
        assert self.registry.detect_asset_class("ETH/BTC") == AssetClass.CRYPTO

    def test_crypto_sol_usdt(self):
        assert self.registry.detect_asset_class("SOL/USDT") == AssetClass.CRYPTO

    def test_forex_eur_usd_slash(self):
        assert self.registry.detect_asset_class("EUR/USD") == AssetClass.FOREX

    def test_forex_gbp_jpy_slash(self):
        assert self.registry.detect_asset_class("GBP/JPY") == AssetClass.FOREX

    def test_forex_six_char(self):
        assert self.registry.detect_asset_class("EURUSD") == AssetClass.FOREX

    def test_forex_audjpy(self):
        assert self.registry.detect_asset_class("AUDJPY") == AssetClass.FOREX

    def test_commodity_gold(self):
        assert self.registry.detect_asset_class("GOLD") == AssetClass.COMMODITY

    def test_commodity_oil(self):
        assert self.registry.detect_asset_class("OIL") == AssetClass.COMMODITY

    def test_commodity_natgas(self):
        assert self.registry.detect_asset_class("NATGAS") == AssetClass.COMMODITY

    def test_commodity_silver(self):
        assert self.registry.detect_asset_class("SILVER") == AssetClass.COMMODITY

    def test_equity_aapl(self):
        assert self.registry.detect_asset_class("AAPL") == AssetClass.EQUITY

    def test_equity_msft(self):
        assert self.registry.detect_asset_class("MSFT") == AssetClass.EQUITY

    def test_equity_lowercase(self):
        assert self.registry.detect_asset_class("aapl") == AssetClass.EQUITY

    def test_commodity_case_insensitive(self):
        assert self.registry.detect_asset_class("gold") == AssetClass.COMMODITY

    def test_forex_case_insensitive(self):
        assert self.registry.detect_asset_class("eurusd") == AssetClass.FOREX

    def test_equity_brk_b(self):
        assert self.registry.detect_asset_class("BRK-B") == AssetClass.EQUITY

    def test_commodity_from_config_proxies(self):
        """Symbols in commodity_etf_proxies should be detected as COMMODITY."""
        assert self.registry.detect_asset_class("AGRICULTURE") == AssetClass.COMMODITY


# =============================================================================
# get_provider Tests
# =============================================================================


class TestGetProvider:
    """Tests for ProviderRegistry.get_provider."""

    def setup_method(self):
        self.registry = ProviderRegistry()

    def test_equity_provider_type(self):
        from quant_tools.data.providers.equity import EquityProvider

        provider = self.registry.get_provider(AssetClass.EQUITY)
        assert isinstance(provider, EquityProvider)

    def test_crypto_provider_type(self):
        from quant_tools.data.providers.crypto import CryptoProvider

        provider = self.registry.get_provider(AssetClass.CRYPTO)
        assert isinstance(provider, CryptoProvider)

    def test_forex_provider_type(self):
        from quant_tools.data.providers.forex import ForexProvider

        provider = self.registry.get_provider(AssetClass.FOREX)
        assert isinstance(provider, ForexProvider)

    def test_commodity_provider_type(self):
        from quant_tools.data.providers.commodity import CommodityProvider

        provider = self.registry.get_provider(AssetClass.COMMODITY)
        assert isinstance(provider, CommodityProvider)

    def test_singleton_pattern(self):
        """Same provider returned for repeated calls."""
        p1 = self.registry.get_provider(AssetClass.EQUITY)
        p2 = self.registry.get_provider(AssetClass.EQUITY)
        assert p1 is p2


# =============================================================================
# fetch_ohlcv (auto-detect) Tests
# =============================================================================


class TestRegistryFetchOhlcv:
    """Tests for ProviderRegistry.fetch_ohlcv with auto-detection."""

    def _sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.bdate_range(start="2023-06-01", periods=20)
        close = 100 * np.cumprod(1 + np.random.normal(0, 0.01, 20))
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 10_000_000, 20),
            },
            index=dates,
        )

    @patch("quant_tools.data.providers.equity.yf")
    def test_auto_detect_equity(self, mock_yf):
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = self._sample_df()

        registry = ProviderRegistry()
        df = registry.fetch_ohlcv("AAPL")
        assert not df.empty

    @patch("quant_tools.data.providers.commodity.yf")
    def test_auto_detect_commodity(self, mock_yf):
        from quant_tools.retry import yfinance_breaker

        yfinance_breaker.reset()
        mock_yf.download.return_value = self._sample_df()

        registry = ProviderRegistry()
        df = registry.fetch_ohlcv("GOLD")
        assert not df.empty
