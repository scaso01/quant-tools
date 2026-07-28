"""Provider registry with auto-detection and delegation."""
from __future__ import annotations

import logging

import pandas as pd

from quant_tools.config import cfg
from quant_tools.models import AssetClass

log = logging.getLogger("quant_tools")

# Known forex currency codes for detection
_FOREX_CURRENCIES = frozenset({
    "EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD",
    "SEK", "NOK", "DKK", "HKD", "SGD", "ZAR", "MXN", "TRY",
})

# Known commodity names for detection
_COMMODITY_NAMES = frozenset({
    "GOLD", "OIL", "SILVER", "NATGAS", "AGRICULTURE",
    "COPPER", "PLATINUM", "WHEAT", "CORN", "SOYBEANS",
    "PALLADIUM", "COTTON", "SUGAR", "COFFEE", "COCOA",
})


class ProviderRegistry:
    """Central registry for asset data providers.

    Auto-detects asset class from symbol patterns and delegates
    to the appropriate provider singleton.
    """

    def __init__(self) -> None:
        self._providers: dict[AssetClass, object] = {}

    def detect_asset_class(self, symbol: str) -> AssetClass:
        """Detect the asset class from a symbol string.

        Rules:
        - "/" in symbol -> CRYPTO (e.g., BTC/USDT)
        - 6-char uppercase with known currencies -> FOREX
        - Known commodity name or in config proxies -> COMMODITY
        - Otherwise -> EQUITY

        Args:
            symbol: Instrument identifier.

        Returns:
            Detected AssetClass.
        """
        upper = symbol.upper().strip()

        # Crypto: contains "/" (e.g., BTC/USDT, ETH/BTC)
        if "/" in upper:
            parts = upper.split("/")
            # Forex pairs also use "/" (EUR/USD) — check if both sides are currencies
            if len(parts) == 2 and parts[0] in _FOREX_CURRENCIES and parts[1] in _FOREX_CURRENCIES:
                return AssetClass.FOREX
            return AssetClass.CRYPTO

        # Commodity: futures tickers, known names, or in config proxies
        if upper.endswith("=F"):
            return AssetClass.COMMODITY
        if upper in _COMMODITY_NAMES or upper in cfg.commodity_etf_proxies:
            return AssetClass.COMMODITY
        if upper in cfg.commodity_etf_proxies.values():
            return AssetClass.COMMODITY

        # Forex: 6-char string where both halves are known currencies
        if len(upper) == 6 and upper[:3] in _FOREX_CURRENCIES and upper[3:] in _FOREX_CURRENCIES:
            return AssetClass.FOREX

        # Default: equity
        return AssetClass.EQUITY

    def get_provider(self, asset_class: AssetClass):
        """Get or create a singleton provider for an asset class.

        Args:
            asset_class: The asset class to get a provider for.

        Returns:
            DataProvider instance.
        """
        if asset_class not in self._providers:
            self._providers[asset_class] = self._create_provider(asset_class)
        return self._providers[asset_class]

    def _create_provider(self, asset_class: AssetClass):
        """Create a new provider instance for an asset class.

        Args:
            asset_class: The asset class.

        Returns:
            DataProvider instance.
        """
        # Lazy imports to avoid circular dependencies
        if asset_class == AssetClass.EQUITY:
            from quant_tools.data.providers.equity import EquityProvider
            return EquityProvider()
        elif asset_class == AssetClass.CRYPTO:
            from quant_tools.data.providers.crypto import CryptoProvider
            return CryptoProvider()
        elif asset_class == AssetClass.FOREX:
            from quant_tools.data.providers.forex import ForexProvider
            return ForexProvider()
        elif asset_class == AssetClass.COMMODITY:
            from quant_tools.data.providers.commodity import CommodityProvider
            return CommodityProvider()
        else:
            from quant_tools.errors import DataError
            raise DataError(f"No provider for asset class: {asset_class}")

    def fetch_ohlcv(
        self, symbol: str, period: str = "1y", **kwargs,
    ) -> pd.DataFrame:
        """Auto-detect asset class and fetch OHLCV data.

        Args:
            symbol: Instrument identifier.
            period: Lookback period.
            **kwargs: Additional arguments passed to the provider.

        Returns:
            DataFrame with Open, High, Low, Close, Volume and DatetimeIndex.
        """
        asset_class = self.detect_asset_class(symbol)
        provider = self.get_provider(asset_class)
        log.debug("Routing %s to %s provider", symbol, asset_class.value)
        return provider.fetch_ohlcv(symbol, period=period, **kwargs)
