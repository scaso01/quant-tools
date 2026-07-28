"""Unified multi-asset data providers."""
from __future__ import annotations

from quant_tools.data.providers.base import DataProvider
from quant_tools.data.providers.equity import EquityProvider
from quant_tools.data.providers.crypto import CryptoProvider
from quant_tools.data.providers.forex import ForexProvider
from quant_tools.data.providers.commodity import CommodityProvider
from quant_tools.data.providers.registry import ProviderRegistry

__all__ = [
    "DataProvider", "EquityProvider", "CryptoProvider",
    "ForexProvider", "CommodityProvider", "ProviderRegistry",
]
