"""yfinance wrapper with caching, retry, and thread safety."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from quant_tools.cache import CacheManager
from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import retry_with_backoff, yfinance_breaker

log = logging.getLogger("quant_tools")

_cache = CacheManager()


@retry_with_backoff(max_attempts=3, initial_delay=2.0, api_name="yfinance")
def fetch_prices(
    symbol: str,
    start: str | date | None = None,
    end: str | date | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """Fetch OHLCV price data for a symbol.

    Checks cache first, falls back to yfinance API.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        start: Start date (YYYY-MM-DD or date object).
        end: End date (YYYY-MM-DD or date object).
        period: yfinance period string (e.g., "1y", "6mo"). Used if start/end not provided.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close.

    Raises:
        DataError: If no data is returned for the symbol.
        NetworkError: If yfinance connection fails.
    """
    if yfinance_breaker.is_open:
        raise NetworkError(f"yfinance circuit breaker is open for {symbol}")

    start_str = str(start) if start else None
    end_str = str(end) if end else None

    # Check cache
    if start_str and end_str:
        cached = _cache.get_prices(symbol, start_str, end_str)
        if cached:
            log.debug("Cache hit for %s (%s to %s)", symbol, start_str, end_str)
            df = pd.DataFrame(cached)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
            return df

    try:
        if start_str and end_str:
            df = yf.download(symbol, start=start_str, end=end_str, progress=False)
        else:
            df = yf.download(symbol, period=period, progress=False)
    except Exception as exc:
        yfinance_breaker.record_failure()
        raise NetworkError(f"yfinance download failed for {symbol}: {exc}") from exc

    if df is None or df.empty:
        yfinance_breaker.record_failure()
        raise DataError(f"No price data returned for {symbol}")

    yfinance_breaker.record_success()

    # Flatten MultiIndex columns (yfinance >= 0.2.40 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Drop duplicate column names that can arise from flattening
    df = df.loc[:, ~df.columns.duplicated()]

    # Cache the data
    try:
        prices_list = []
        for idx, row in df.iterrows():
            prices_list.append({
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": int(row.get("Volume", 0)),
                "adj_close": float(row["Adj Close"]) if "Adj Close" in row else None,
            })
        _cache.store_prices(symbol, prices_list)
    except Exception:
        log.debug("Failed to cache prices for %s", symbol, exc_info=True)

    return df


def fetch_ticker_info(symbol: str) -> dict[str, object]:
    """Fetch ticker metadata (sector, market cap, PE, etc.).

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict of ticker info fields.

    Raises:
        DataError: If ticker info cannot be retrieved.
    """
    cached = _cache.get_metadata(symbol)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as exc:
        raise DataError(f"Failed to fetch info for {symbol}: {exc}") from exc

    if not info:
        raise DataError(f"No info returned for {symbol}")

    result = {
        "symbol": symbol.upper(),
        "name": info.get("longName", info.get("shortName", symbol)),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "pb_ratio": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "debt_to_equity": info.get("debtToEquity"),
    }

    try:
        _cache.store_metadata(symbol, result)
    except Exception:
        log.debug("Failed to cache metadata for %s", symbol, exc_info=True)

    return result


def get_cache_manager() -> CacheManager:
    """Return the global cache manager for direct access."""
    return _cache
