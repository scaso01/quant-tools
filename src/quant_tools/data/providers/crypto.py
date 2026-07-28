"""Crypto data provider (ccxt)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_tools.config import cfg
from quant_tools.data.providers.base import DataProvider
from quant_tools.errors import DataError, NetworkError
from quant_tools.models import AssetClass
from quant_tools.retry import ccxt_breaker

log = logging.getLogger("quant_tools")

# Period string to days mapping
_PERIOD_DAYS: dict[str, int] = {
    "max": 3650,
    "10y": 3650,
    "5y": 1825,
    "2y": 730,
    "1y": 365,
    "6mo": 180,
    "3mo": 90,
    "1mo": 30,
    "1w": 7,
    "5d": 5,
    "1d": 1,
}

# Interval string to ccxt timeframe mapping
_INTERVAL_MAP: dict[str, str] = {
    "1mo": "1M",
    "1wk": "1w",
    "1d": "1d",
    "12h": "12h",
    "8h": "8h",
    "6h": "6h",
    "4h": "4h",
    "1h": "1h",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}


class CryptoProvider(DataProvider):
    """Provides crypto data via ccxt exchanges."""

    def __init__(self, exchange_id: str | None = None) -> None:
        self._exchange_id = exchange_id or cfg.crypto_default_exchange
        self._exchange = None

    @property
    def name(self) -> str:
        return "crypto"

    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.CRYPTO]

    def _get_exchange(self):
        """Lazy-init the ccxt exchange instance."""
        if self._exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_id, None)
            if exchange_class is None:
                raise DataError(f"Unknown ccxt exchange: {self._exchange_id}")
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch crypto OHLCV data from ccxt.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            period: Lookback period.
            interval: Bar interval.

        Returns:
            DataFrame with Open, High, Low, Close, Volume and DatetimeIndex.

        Raises:
            DataError: If no data is returned or symbol is invalid.
            NetworkError: If the circuit breaker is open.
        """
        if ccxt_breaker.is_open:
            raise NetworkError(f"ccxt circuit breaker is open for {symbol}")

        days = _PERIOD_DAYS.get(period, 365)
        since_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
        since_ms = int(since_dt.timestamp() * 1000)

        timeframe = _INTERVAL_MAP.get(interval, "1d")

        try:
            exchange = self._get_exchange()
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000,
            )
        except Exception as exc:
            ccxt_breaker.record_failure()
            raise DataError(f"ccxt fetch failed for {symbol}: {exc}") from exc

        if not ohlcv:
            ccxt_breaker.record_failure()
            raise DataError(f"No crypto data returned for {symbol}")

        ccxt_breaker.record_success()

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.index.name = None

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_info(self, symbol: str) -> dict:
        """Fetch crypto pair metadata from exchange markets.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").

        Returns:
            Dict of metadata fields.

        Raises:
            DataError: If the symbol is not found on the exchange.
        """
        try:
            exchange = self._get_exchange()
            exchange.load_markets()
            market = exchange.markets.get(symbol)
        except Exception as exc:
            raise DataError(f"Failed to fetch crypto info for {symbol}: {exc}") from exc

        if market is None:
            raise DataError(f"Symbol {symbol} not found on {self._exchange_id}")

        return {
            "symbol": symbol,
            "base": market.get("base", ""),
            "quote": market.get("quote", ""),
            "exchange": self._exchange_id,
            "active": market.get("active", True),
            "asset_class": AssetClass.CRYPTO.value,
        }
