"""Equity data provider (yfinance + stooq + Finnhub fallback)."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from quant_tools.data.providers.base import DataProvider
from quant_tools.data.providers.finnhub_provider import _finnhub_info, _finnhub_ohlcv
from quant_tools.errors import DataError, NetworkError
from quant_tools.models import AssetClass
from quant_tools.retry import finnhub_breaker, yfinance_breaker

log = logging.getLogger("quant_tools")


class EquityProvider(DataProvider):
    """Provides equity data via yfinance with stooq fallback."""

    @property
    def name(self) -> str:
        return "equity"

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.EQUITY]

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch equity OHLCV data.

        Tries yfinance first, falls back to pandas-datareader stooq.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
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

        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
        except Exception as exc:
            yfinance_breaker.record_failure()
            log.warning("yfinance failed for %s, trying stooq fallback: %s", symbol, exc)
            df = self._stooq_fallback(symbol)

        if df is None or df.empty:
            yfinance_breaker.record_failure()
            # Try Finnhub as third fallback
            if not finnhub_breaker.is_open:
                log.info("Trying Finnhub fallback for %s", symbol)
                df = _finnhub_ohlcv(symbol, period, interval)
                if df is not None and not df.empty:
                    finnhub_breaker.record_success()
                else:
                    finnhub_breaker.record_failure()

        if df is None or df.empty:
            raise DataError(f"No equity data returned for {symbol}")

        yfinance_breaker.record_success()

        # Flatten MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

        # Normalize column names
        col_map = {
            "Adj Close": "Adj_Close",
            "adj close": "Adj_Close",
        }
        df = df.rename(columns=col_map)

        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in df.columns:
                raise DataError(f"Missing column {col} in equity data for {symbol}")

        return df[expected]

    def _stooq_fallback(self, symbol: str) -> pd.DataFrame | None:
        """Attempt to fetch data from stooq via pandas-datareader.

        Args:
            symbol: Ticker symbol.

        Returns:
            DataFrame or None on failure.
        """
        try:
            from pandas_datareader import data as pdr

            df = pdr.DataReader(symbol, "stooq")
            if df is not None and not df.empty:
                df = df.sort_index()
                return df
        except Exception as exc:
            log.warning("stooq fallback also failed for %s: %s", symbol, exc)
        return None

    def fetch_info(self, symbol: str) -> dict:
        """Fetch equity ticker metadata.

        Tries yfinance first, falls back to Finnhub if yfinance
        returns empty or minimal info.

        Args:
            symbol: Ticker symbol.

        Returns:
            Dict of metadata fields.

        Raises:
            DataError: If info cannot be retrieved from any source.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as exc:
            log.warning("yfinance info failed for %s: %s", symbol, exc)
            info = {}

        # If yfinance returned nothing useful, try Finnhub
        if not info or not info.get("longName"):
            fb_info = _finnhub_info(symbol)
            if fb_info:
                return {
                    "symbol": symbol.upper(),
                    "name": fb_info.get("longName", symbol),
                    "sector": fb_info.get("sector", ""),
                    "industry": fb_info.get("industry", ""),
                    "market_cap": fb_info.get("marketCap"),
                    "pe_ratio": fb_info.get("trailingPE"),
                    "dividend_yield": fb_info.get("dividendYield"),
                    "asset_class": AssetClass.EQUITY.value,
                }

        if not info:
            raise DataError(f"No info returned for {symbol}")

        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "asset_class": AssetClass.EQUITY.value,
        }
