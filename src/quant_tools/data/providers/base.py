"""Abstract base class for all data providers."""
from __future__ import annotations

import abc
import logging

import pandas as pd

from quant_tools.models import AssetClass

log = logging.getLogger("quant_tools")


class DataProvider(abc.ABC):
    """Base class for all asset data providers.

    Subclasses must implement fetch_ohlcv, fetch_info, and
    supported_asset_classes.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Instrument identifier.
            period: Lookback period (e.g., "1y", "6mo", "3mo").
            interval: Bar interval (e.g., "1d", "1h").

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            and a DatetimeIndex.
        """

    @abc.abstractmethod
    def fetch_info(self, symbol: str) -> dict:
        """Fetch metadata for a symbol.

        Args:
            symbol: Instrument identifier.

        Returns:
            Dict of metadata fields.
        """

    @abc.abstractmethod
    def supported_asset_classes(self) -> list[AssetClass]:
        """Return asset classes this provider can handle."""
