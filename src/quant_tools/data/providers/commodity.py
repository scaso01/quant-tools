"""Commodity data provider (futures + ETF proxies via yfinance)."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from quant_tools.config import cfg
from quant_tools.data.providers.base import DataProvider
from quant_tools.errors import DataError, NetworkError
from quant_tools.models import AssetClass, RollAdjustment, RollMethod
from quant_tools.retry import yfinance_breaker

log = logging.getLogger("quant_tools")


class CommodityProvider(DataProvider):
    """Provides commodity data via futures tickers and ETF proxies."""

    @property
    def name(self) -> str:
        return "commodity"

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.COMMODITY]

    def _resolve_symbol(self, symbol: str) -> str:
        """Map a commodity name to its futures or ETF ticker.

        Args:
            symbol: Commodity name, futures ticker (e.g. "GC=F"), or ETF ticker.

        Returns:
            Resolved ticker symbol.

        Raises:
            DataError: If no commodity symbol mapping is known.
        """
        upper = symbol.upper()
        # Futures tickers pass through directly
        if "=F" in upper:
            return upper
        mapped = cfg.commodity_etf_proxies.get(upper)
        if mapped:
            return mapped
        # If the symbol itself looks like a ticker (short, uppercase), use it directly
        if len(upper) <= 5 and upper.isalpha():
            return upper
        raise DataError(
            f"No commodity symbol for {symbol!r}. "
            f"Known: {list(cfg.commodity_etf_proxies.keys())}"
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch commodity OHLCV data via futures or ETF proxy.

        Args:
            symbol: Commodity name (e.g., "GOLD"), futures ticker (e.g., "GC=F"),
                or ETF ticker (e.g., "GLD").
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

        ticker = self._resolve_symbol(symbol)

        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as exc:
            yfinance_breaker.record_failure()
            raise DataError(
                f"Failed to fetch commodity data for {symbol} ({ticker}): {exc}"
            ) from exc

        if df is None or df.empty:
            yfinance_breaker.record_failure()
            raise DataError(f"No commodity data returned for {symbol} ({ticker})")

        yfinance_breaker.record_success()

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in df.columns:
                raise DataError(f"Missing column {col} in commodity data for {symbol}")

        return df[expected]

    def fetch_info(self, symbol: str) -> dict:
        """Fetch commodity metadata.

        Includes futures/ETF info and optional COT data.

        Args:
            symbol: Commodity name, futures ticker, or ETF ticker.

        Returns:
            Dict of metadata fields.
        """
        upper = symbol.upper()
        resolved = self._resolve_symbol(upper)

        info: dict = {
            "symbol": upper,
            "ticker": resolved,
            "asset_class": AssetClass.COMMODITY.value,
        }

        # Try to get info from yfinance
        try:
            yf_ticker = yf.Ticker(resolved)
            yf_info = yf_ticker.info or {}
            info["name"] = yf_info.get("longName", "")
            info["expense_ratio"] = yf_info.get("annualReportExpenseRatio")
        except Exception:
            log.debug("Could not fetch yfinance info for %s", resolved)

        # Optional COT data (graceful failure)
        info["cot_data"] = self._fetch_cot(upper)

        return info

    def fetch_continuous(
        self,
        root: str,
        period: str = "1y",
        method: RollMethod = RollMethod.CALENDAR,
        adjustment: RollAdjustment = RollAdjustment.PANAMA,
    ) -> pd.DataFrame:
        """Fetch back-adjusted continuous futures series.

        Args:
            root: Commodity root symbol (e.g., "GC", "CL").
            period: Lookback period (e.g., "1y", "6mo").
            method: Roll detection method.
            adjustment: Back-adjustment method.

        Returns:
            DataFrame with Open, High, Low, Close, Volume and DatetimeIndex.

        Raises:
            DataError: If no data is returned.
            NetworkError: If the circuit breaker is open.
        """
        from datetime import date, timedelta

        from quant_tools.data.futures import build_continuous_series

        if yfinance_breaker.is_open:
            raise NetworkError(f"yfinance circuit breaker is open for {root}")

        # Convert period string to start/end dates
        period_map = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
            "2y": 730, "5y": 1825, "10y": 3650,
        }
        days = period_map.get(period, 365)
        end = date.today()
        start = end - timedelta(days=days)

        try:
            df = build_continuous_series(
                root=root, start=start, end=end,
                method=method, adjustment=adjustment,
            )
        except Exception as exc:
            yfinance_breaker.record_failure()
            raise DataError(
                f"Failed to build continuous series for {root}: {exc}"
            ) from exc

        if df is None or df.empty:
            yfinance_breaker.record_failure()
            raise DataError(f"No continuous series data for {root}")

        yfinance_breaker.record_success()
        return df

    def _fetch_cot(self, commodity: str) -> dict | None:
        """Attempt to fetch Commitment of Traders data.

        Args:
            commodity: Commodity name.

        Returns:
            COT summary dict or None.
        """
        try:
            import cot_reports

            df = cot_reports.cot_year(2024)
            if df is not None and not df.empty:
                # Filter for relevant commodity if possible
                return {"available": True, "rows": len(df)}
        except Exception:
            log.debug("COT data not available for %s", commodity)
        return None
