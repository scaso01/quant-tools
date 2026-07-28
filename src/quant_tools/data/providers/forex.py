"""Forex data provider (yfinance + ccxt fallback)."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from quant_tools.data.providers.base import DataProvider
from quant_tools.errors import DataError, NetworkError
from quant_tools.models import AssetClass
from quant_tools.retry import yfinance_breaker

log = logging.getLogger("quant_tools")


def _to_yf_forex_symbol(pair: str) -> str:
    """Convert forex pair to yfinance format.

    "EUR/USD" -> "EURUSD=X"
    "EURUSD"  -> "EURUSD=X"

    Args:
        pair: Forex pair string.

    Returns:
        yfinance-compatible symbol.
    """
    clean = pair.replace("/", "").replace(" ", "").upper()
    if not clean.endswith("=X"):
        clean += "=X"
    return clean


class ForexProvider(DataProvider):
    """Provides forex data via yfinance with ccxt fallback."""

    @property
    def name(self) -> str:
        return "forex"

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.FOREX]

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch forex OHLCV data.

        Args:
            symbol: Forex pair (e.g., "EUR/USD").
            period: Lookback period.
            interval: Bar interval.

        Returns:
            DataFrame with Open, High, Low, Close, Volume and DatetimeIndex.

        Raises:
            DataError: If no data is returned.
            NetworkError: If the circuit breaker is open.
        """
        if yfinance_breaker.is_open:
            raise NetworkError(f"yfinance circuit breaker is open for {symbol}")

        yf_symbol = _to_yf_forex_symbol(symbol)

        try:
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        except Exception as exc:
            yfinance_breaker.record_failure()
            log.warning("yfinance failed for forex %s, trying ccxt fallback: %s", symbol, exc)
            df = self._ccxt_fallback(symbol, period, interval)

        if df is None or df.empty:
            yfinance_breaker.record_failure()
            raise DataError(f"No forex data returned for {symbol}")

        yfinance_breaker.record_success()

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in df.columns:
                # Forex volume is often 0; ensure column exists
                if col == "Volume":
                    df["Volume"] = 0
                else:
                    raise DataError(f"Missing column {col} in forex data for {symbol}")

        return df[expected]

    def _ccxt_fallback(
        self, symbol: str, period: str, interval: str,
    ) -> pd.DataFrame | None:
        """Attempt to fetch forex data from ccxt for exotic pairs.

        Args:
            symbol: Forex pair.
            period: Lookback period.
            interval: Bar interval.

        Returns:
            DataFrame or None on failure.
        """
        try:
            from quant_tools.data.providers.crypto import CryptoProvider

            provider = CryptoProvider(exchange_id="kraken")
            return provider.fetch_ohlcv(symbol, period=period, interval=interval)
        except Exception as exc:
            log.warning("ccxt forex fallback failed for %s: %s", symbol, exc)
        return None

    def fetch_info(self, symbol: str) -> dict:
        """Fetch basic forex pair metadata.

        Args:
            symbol: Forex pair (e.g., "EUR/USD").

        Returns:
            Dict of metadata fields.
        """
        clean = symbol.replace(" ", "").upper()
        parts = clean.split("/") if "/" in clean else [clean[:3], clean[3:6]]
        base = parts[0] if len(parts) >= 1 else ""
        quote = parts[1] if len(parts) >= 2 else ""

        return {
            "symbol": symbol.upper(),
            "base_currency": base,
            "quote_currency": quote,
            "pair": f"{base}/{quote}",
            "asset_class": AssetClass.FOREX.value,
        }
