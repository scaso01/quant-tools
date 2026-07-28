"""Pytest fixtures for quant-tools tests.

Provides:
- Network guards: Block all real HTTP requests
- yfinance guard: Mock yf.download() and yf.Ticker()
- FRED guard: Mock fredapi.Fred
- Log isolation: NullHandler swap
- Sample data fixtures: price DataFrames, portfolios, fundamentals
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Network Guard — prevent real HTTP requests during tests
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_real_network():
    """Prevent real HTTP requests during tests."""

    def _blocked_request(*args, **kwargs):
        raise RuntimeError(
            "Real HTTP request blocked by conftest.py! "
            "Use unittest.mock to mock HTTP calls in tests."
        )

    patches = []

    # Block requests library
    try:
        import requests as req_lib
        for method in ("get", "post", "put", "delete", "patch", "head"):
            p = patch.object(req_lib, method, side_effect=_blocked_request)
            patches.append(p)
    except ImportError:
        pass

    # Block httpx
    try:
        import httpx
        p = patch.object(httpx, "get", side_effect=_blocked_request)
        patches.append(p)
        p = patch.object(httpx, "post", side_effect=_blocked_request)
        patches.append(p)
    except ImportError:
        pass

    # Block urllib
    try:
        import urllib.request
        p = patch.object(urllib.request, "urlopen", side_effect=_blocked_request)
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()


# =============================================================================
# yfinance Guard — prevent real Yahoo Finance API calls
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_yfinance():
    """Prevent real yfinance API calls during tests."""
    patches = []

    try:
        import yfinance as yf
        p = patch.object(yf, "download", side_effect=RuntimeError(
            "Real yf.download() blocked by conftest.py!"
        ))
        patches.append(p)
        p = patch.object(yf, "Ticker", side_effect=RuntimeError(
            "Real yf.Ticker() blocked by conftest.py!"
        ))
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()


# =============================================================================
# FRED Guard — prevent real FRED API calls
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_fred():
    """Prevent real FRED API calls during tests."""
    patches = []

    try:
        import fredapi
        p = patch.object(fredapi, "Fred", side_effect=RuntimeError(
            "Real fredapi.Fred() blocked by conftest.py!"
        ))
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()


# =============================================================================
# CCXT Guard — prevent real crypto exchange calls
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_ccxt():
    """Prevent real CCXT exchange API calls during tests."""
    patches = []

    try:
        import ccxt
        # Block the two most common exchange constructors
        for exchange_name in ("kraken", "binance", "coinbase"):
            cls = getattr(ccxt, exchange_name, None)
            if cls is not None:
                p = patch.object(
                    cls, "fetch_ohlcv",
                    side_effect=RuntimeError(f"Real ccxt.{exchange_name}.fetch_ohlcv() blocked!"),
                )
                patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# =============================================================================
# WebSocket Guard — prevent real WebSocket connections
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_websockets():
    """Prevent real WebSocket connections during tests."""
    patches = []

    try:
        import websockets
        p = patch.object(
            websockets, "connect",
            side_effect=RuntimeError("Real websockets.connect() blocked!"),
        )
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# =============================================================================
# Alpaca Guard — prevent real Alpaca API calls
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _block_alpaca():
    """Prevent real Alpaca Trade API calls during tests."""
    patches = []

    try:
        import alpaca_trade_api
        p = patch.object(
            alpaca_trade_api, "REST",
            side_effect=RuntimeError("Real alpaca_trade_api.REST() blocked!"),
        )
        patches.append(p)
    except ImportError:
        pass

    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# =============================================================================
# Log Isolation
# =============================================================================


@pytest.fixture(autouse=True, scope="session")
def _isolate_log_handlers():
    """Replace quant_tools file handlers with NullHandler during tests."""
    qt_logger = logging.getLogger("quant_tools")
    original_handlers = qt_logger.handlers[:]

    qt_logger.handlers = [logging.NullHandler()]

    yield

    qt_logger.handlers = original_handlers


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate a realistic sample price DataFrame (252 trading days)."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range(start="2023-01-02", periods=n)

    # Random walk from 100
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_prices = (high + low) / 2
    volume = np.random.randint(1_000_000, 50_000_000, n)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=dates,
    )


@pytest.fixture
def sample_close_array(sample_prices: pd.DataFrame) -> np.ndarray:
    """Extract close prices as numpy array."""
    return sample_prices["Close"].values


@pytest.fixture
def sample_portfolio_weights() -> dict[str, float]:
    """Sample portfolio weights."""
    return {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "BND": 0.30,
        "GLD": 0.15,
        "VNQ": 0.10,
    }


@pytest.fixture
def sample_ticker_info() -> dict[str, object]:
    """Sample ticker info dict."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_000_000_000_000,
        "pe_ratio": 30.5,
        "forward_pe": 28.0,
        "dividend_yield": 0.005,
        "beta": 1.2,
        "52w_high": 200.0,
        "52w_low": 140.0,
        "avg_volume": 50_000_000,
        "pb_ratio": 45.0,
        "roe": 0.15,
        "profit_margin": 0.25,
        "revenue_growth": 0.08,
        "debt_to_equity": 1.5,
    }


@pytest.fixture
def sample_multi_prices() -> dict[str, pd.DataFrame]:
    """Generate price DataFrames for multiple tickers."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range(start="2023-01-02", periods=n)
    tickers = {}

    for i, symbol in enumerate(["AAPL", "MSFT", "BND", "GLD", "VNQ"]):
        np.random.seed(42 + i)
        returns = np.random.normal(0.0003 + i * 0.0001, 0.01 + i * 0.002, n)
        close = 100 * np.cumprod(1 + returns)
        tickers[symbol] = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 50_000_000, n),
                "Adj Close": close,
            },
            index=dates,
        )

    return tickers


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_cache.db")


