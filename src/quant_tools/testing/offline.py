"""Deterministic offline market data, for CI and for reproducible screenshots.

The e2e suite drives a real browser against a real Streamlit server, so the
in-process network guards in ``tests/conftest.py`` do not apply to it: the
server is a separate process and happily calls Yahoo Finance. That makes the
suite unrunnable on a CI box and flaky everywhere else, since the upstream feed
throttles after a few dozen requests.

Setting ``QUANT_TOOLS_OFFLINE=1`` before the dashboard starts replaces
``yfinance.download`` and ``yfinance.Ticker`` with a seeded generator. Prices
are synthetic but stable: the same symbol always produces the same series, so
assertions on chart contents and metric cards stay meaningful.

**This fabricates prices.** Nothing here should ever run against a real
analysis. ``quant_tools.dashboard.app`` renders a permanent banner whenever the
flag is on, and :func:`is_active` is public so any other entry point can do the
same.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("quant_tools")

ENV_VAR = "QUANT_TOOLS_OFFLINE"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

# Trading days per calendar unit, close enough for generating a window.
_PERIOD_DAYS: dict[str, int] = {
    "1d": 1, "5d": 5, "1mo": 21, "3mo": 63, "6mo": 126,
    "1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 2520,
}

_SECTORS = (
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Energy", "Industrials", "Utilities", "Communication Services",
)


def is_active() -> bool:
    """True when offline mode is switched on via the environment."""
    return os.environ.get(ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _seed(symbol: str) -> int:
    """Stable per-symbol seed.

    ``hash()`` is salted per interpreter, so it would give a different series on
    every server restart and break any assertion on a specific value.
    """
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _resolve_window(
    start: Any = None, end: Any = None, period: str = "1y"
) -> tuple[pd.Timestamp, pd.Timestamp, int | None]:
    """Turn yfinance's start/end/period arguments into a concrete date range.

    The third element is a session cap. A period asks for a count of trading
    days, not a span of calendar days, so the caller over-generates and trims;
    padding the calendar range alone overshoots badly on short periods.
    """
    if start is not None and end is not None:
        return pd.Timestamp(start), pd.Timestamp(end), None

    finish = pd.Timestamp(end) if end is not None else pd.Timestamp(date.today())
    if start is not None:
        return pd.Timestamp(start), finish, None

    key = str(period).lower()
    if key == "ytd":
        return pd.Timestamp(year=finish.year, month=1, day=1), finish, None
    sessions = _PERIOD_DAYS.get(key, 252)
    return finish - timedelta(days=int(sessions * 1.6) + 10), finish, sessions


def generate_prices(
    symbol: str, start: Any = None, end: Any = None, period: str = "1y"
) -> pd.DataFrame:
    """Build a deterministic OHLCV frame for *symbol*.

    A geometric random walk, seeded from the symbol, so every column holds
    plausible values and the usual invariants hold: high is the largest of the
    four prices, low the smallest, volume positive.
    """
    first, last, cap = _resolve_window(start, end, period)
    index = pd.bdate_range(start=first, end=last)
    if cap is not None:
        index = index[-cap:]
    if len(index) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    rng = np.random.default_rng(_seed(symbol))
    n = len(index)

    # Opening price band keeps penny stocks and index-priced names apart.
    base = 20.0 + (_seed(symbol) % 380)
    returns = rng.normal(loc=0.0003, scale=0.015, size=n)
    close = base * np.exp(np.cumsum(returns))

    prev_close = np.concatenate(([base], close[:-1]))
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.004, size=n))
    span = np.abs(rng.normal(0.0, 0.008, size=n)) + 0.002
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    volume = rng.lognormal(mean=15.0, sigma=0.5, size=n).astype("int64")

    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=index,
    )
    frame.index.name = "Date"
    return frame


def generate_info(symbol: str) -> dict[str, Any]:
    """Build deterministic ticker metadata for *symbol*."""
    rng = np.random.default_rng(_seed(symbol))
    prices = generate_prices(symbol, period="1y")
    closes = prices["Close"]
    last = float(closes.iloc[-1]) if len(closes) else 100.0
    return {
        "longName": f"{symbol.upper()} Test Corp",
        "shortName": symbol.upper(),
        "sector": _SECTORS[_seed(symbol) % len(_SECTORS)],
        "industry": "Offline Fixture",
        "marketCap": int(last * rng.integers(10_000_000, 2_000_000_000)),
        "trailingPE": round(float(rng.uniform(8, 45)), 2),
        "forwardPE": round(float(rng.uniform(8, 40)), 2),
        "dividendYield": round(float(rng.uniform(0, 0.05)), 4),
        "beta": round(float(rng.uniform(0.4, 2.0)), 2),
        "fiftyTwoWeekHigh": round(float(closes.max()) if len(closes) else last, 2),
        "fiftyTwoWeekLow": round(float(closes.min()) if len(closes) else last, 2),
        "averageVolume": int(prices["Volume"].mean()) if len(prices) else 1_000_000,
        "priceToBook": round(float(rng.uniform(0.8, 12.0)), 2),
        "returnOnEquity": round(float(rng.uniform(-0.1, 0.45)), 4),
        "profitMargins": round(float(rng.uniform(-0.05, 0.35)), 4),
        "revenueGrowth": round(float(rng.uniform(-0.2, 0.6)), 4),
        "debtToEquity": round(float(rng.uniform(0, 250)), 2),
        "regularMarketPrice": round(last, 2),
        "currency": "USD",
    }


class _OfflineTicker:
    """Stand-in for ``yfinance.Ticker`` covering the attributes this app reads."""

    def __init__(self, symbol: str, *args: Any, **kwargs: Any) -> None:
        self.ticker = str(symbol).upper()

    @property
    def info(self) -> dict[str, Any]:
        return generate_info(self.ticker)

    @property
    def fast_info(self) -> dict[str, Any]:
        return generate_info(self.ticker)

    @property
    def news(self) -> list[dict[str, Any]]:
        stamp = datetime(2026, 1, 2, 14, 30).isoformat()
        return [
            {
                "id": f"offline-{self.ticker}-{i}",
                "content": {
                    "title": f"{self.ticker} offline fixture headline {i}",
                    "pubDate": stamp,
                    "provider": {"displayName": "Offline Fixture"},
                    "canonicalUrl": {"url": "https://example.invalid/offline"},
                },
            }
            for i in range(1, 4)
        ]

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return generate_prices(
            self.ticker,
            start=kwargs.get("start"),
            end=kwargs.get("end"),
            period=kwargs.get("period", "1y"),
        )


def _download(tickers: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Stand-in for ``yfinance.download``.

    Mirrors the real signature closely enough for this codebase: a single symbol
    gives flat columns, several give the MultiIndex layout callers flatten.
    """
    if isinstance(tickers, str):
        symbols = tickers.split()
    else:
        symbols = [str(t) for t in tickers]

    start = kwargs.get("start", args[0] if args else None)
    end = kwargs.get("end", args[1] if len(args) > 1 else None)
    period = kwargs.get("period", "1y")

    if len(symbols) == 1:
        return generate_prices(symbols[0], start=start, end=end, period=period)

    frames = {
        sym: generate_prices(sym, start=start, end=end, period=period)
        for sym in symbols
    }
    return pd.concat(frames.values(), axis=1, keys=frames.keys()).swaplevel(axis=1)


class _NullCache:
    """A cache that never hits and never stores.

    ``fetch_prices`` consults the on-disk SQLite cache before yfinance, so
    patching the download function alone is not enough: a box that has run the
    app for real still serves genuine (and possibly truncated) rows, and the
    run stops being deterministic. Offline mode has to bypass it.
    """

    def get_prices(self, *args: Any, **kwargs: Any) -> None:
        return None

    def store_prices(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_metadata(self, *args: Any, **kwargs: Any) -> None:
        return None

    def store_metadata(self, *args: Any, **kwargs: Any) -> None:
        return None


def _offline_crypto_ohlcv(
    self: Any, symbol: str, period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """Stand in for the ccxt round trip, with the same generated bars.

    Crypto does not go through yfinance, so patching that alone left the
    exchange call live: the dashboard's crypto scans were reaching Kraken from
    CI, which is exactly what this module exists to prevent. Columns match what
    :class:`~quant_tools.data.providers.crypto.CryptoProvider` returns.
    """
    frame = generate_prices(symbol, period=period)
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def activate(force: bool = False) -> bool:
    """Patch yfinance, ccxt and the price cache with the offline generator.

    Returns True if the patch was applied. Without *force* this is a no-op
    unless :func:`is_active` says the environment asked for it, so importing the
    module can never silently fake a real user's prices.
    """
    if not (force or is_active()):
        return False

    import yfinance as yf

    from quant_tools.data import market_data
    from quant_tools.data.providers import crypto as crypto_provider

    yf.download = _download  # type: ignore[assignment]
    yf.Ticker = _OfflineTicker  # type: ignore[assignment]
    crypto_provider.CryptoProvider.fetch_ohlcv = _offline_crypto_ohlcv  # type: ignore[assignment]
    market_data._cache = _NullCache()  # type: ignore[assignment]
    log.warning(
        "%s is set: market data is SYNTHETIC. Prices, fundamentals and news "
        "are generated, not fetched.",
        ENV_VAR,
    )
    return True
