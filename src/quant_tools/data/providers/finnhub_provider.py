"""Finnhub helper for equity data fallback.

Standalone helper (not a full DataProvider) used by EquityProvider
as a third fallback after yfinance and stooq.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Period string to timedelta mapping
_PERIOD_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
    "5y": 1825, "10y": 3650, "max": 7300,
}

# Interval string to Finnhub resolution mapping
_INTERVAL_MAP = {
    "1d": "D", "1wk": "W", "1mo": "M",
    "1h": "60", "4h": "60",  # Finnhub free tier: 1-min to daily
}


def _finnhub_ohlcv(
    symbol: str, period: str = "1y", interval: str = "1d",
) -> pd.DataFrame | None:
    """Fetch OHLCV from Finnhub. Returns None on any failure."""
    try:
        from quant_tools.config import cfg

        if not cfg.finnhub_api_key:
            return None

        import finnhub

        client = finnhub.Client(api_key=cfg.finnhub_api_key)
        resolution = _INTERVAL_MAP.get(interval, "D")
        days = _PERIOD_DAYS.get(period, 365)
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=days)).timestamp())

        candles = client.stock_candles(symbol, resolution, start, end)
        if not candles or candles.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "Open": candles["o"],
            "High": candles["h"],
            "Low": candles["l"],
            "Close": candles["c"],
            "Volume": candles["v"],
        }, index=pd.to_datetime(candles["t"], unit="s"))
        df.index.name = "Date"
        return df

    except Exception:
        logger.debug("Finnhub OHLCV failed for %s", symbol, exc_info=True)
        return None


def _finnhub_info(symbol: str) -> dict | None:
    """Fetch company profile from Finnhub. Returns None on any failure."""
    try:
        from quant_tools.config import cfg

        if not cfg.finnhub_api_key:
            return None

        import finnhub

        client = finnhub.Client(api_key=cfg.finnhub_api_key)
        profile = client.company_profile2(symbol=symbol)
        if not profile:
            return None

        return {
            "longName": profile.get("name", symbol),
            "sector": profile.get("finnhubIndustry", ""),
            "industry": profile.get("finnhubIndustry", ""),
            "marketCap": profile.get("marketCapitalization", 0) * 1e6,
            "trailingPE": None,
            "dividendYield": None,
        }

    except Exception:
        logger.debug("Finnhub info failed for %s", symbol, exc_info=True)
        return None
