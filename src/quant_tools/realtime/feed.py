"""WebSocket feed manager — connect to market data streams.

Provides TickEvent (normalized tick), FeedConfig (connection settings),
and WebSocketFeedManager (async manager with callbacks).
All external connections go through websockets library — easily mocked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data types
# =============================================================================


class FeedSource(Enum):
    """Supported feed sources."""
    ALPACA = "alpaca"
    BINANCE = "binance"
    POLYGON = "polygon"
    CUSTOM = "custom"


@dataclass
class TickEvent:
    """A single normalized market tick."""

    symbol: str
    price: float
    volume: float
    timestamp: datetime
    source: FeedSource = FeedSource.CUSTOM
    bid: Optional[float] = None
    ask: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread if both sides available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid(self) -> Optional[float]:
        """Mid price if both sides available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None


@dataclass
class FeedConfig:
    """Configuration for a feed connection."""

    source: FeedSource = FeedSource.CUSTOM
    url: str = ""
    symbols: list[str] = field(default_factory=list)
    api_key: str = ""
    api_secret: str = ""
    reconnect_delay: float = 5.0
    max_reconnects: int = 10
    heartbeat_interval: float = 30.0


# =============================================================================
# Normalizers — convert raw exchange messages to TickEvent
# =============================================================================


def normalize_alpaca(raw: dict[str, Any]) -> Optional[TickEvent]:
    """Normalize an Alpaca trade message."""
    try:
        return TickEvent(
            symbol=raw.get("S", raw.get("symbol", "")),
            price=float(raw.get("p", raw.get("price", 0))),
            volume=float(raw.get("s", raw.get("size", 0))),
            timestamp=datetime.fromisoformat(
                raw.get("t", raw.get("timestamp", datetime.now().isoformat()))
            ),
            source=FeedSource.ALPACA,
        )
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Alpaca normalize failed: %s", exc)
        return None


def normalize_binance(raw: dict[str, Any]) -> Optional[TickEvent]:
    """Normalize a Binance trade stream message."""
    try:
        return TickEvent(
            symbol=raw.get("s", ""),
            price=float(raw.get("p", 0)),
            volume=float(raw.get("q", 0)),
            timestamp=datetime.fromtimestamp(raw.get("T", 0) / 1000),
            source=FeedSource.BINANCE,
        )
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Binance normalize failed: %s", exc)
        return None


NORMALIZERS: dict[FeedSource, Callable[[dict[str, Any]], Optional[TickEvent]]] = {
    FeedSource.ALPACA: normalize_alpaca,
    FeedSource.BINANCE: normalize_binance,
}


# =============================================================================
# Feed manager
# =============================================================================


TickCallback = Callable[[TickEvent], None]


class WebSocketFeedManager:
    """Manage WebSocket feed connections and dispatch ticks to callbacks.

    This class manages connection state and callback dispatch.
    Actual WebSocket I/O is delegated to ``websockets`` (or similar),
    which should be injected or mocked in tests.
    """

    def __init__(self, config: FeedConfig) -> None:
        self.config = config
        self._callbacks: list[TickCallback] = []
        self._running: bool = False
        self._reconnect_count: int = 0
        self._last_tick: Optional[TickEvent] = None

    # ── Callback registration ──

    def on_tick(self, callback: TickCallback) -> None:
        """Register a callback to receive tick events."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug("Registered tick callback: %s", callback.__name__)

    def remove_callback(self, callback: TickCallback) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # ── Tick dispatch ──

    def dispatch(self, tick: TickEvent) -> None:
        """Send a tick to all registered callbacks."""
        self._last_tick = tick
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:
                logger.exception("Callback %s raised an exception", cb.__name__)

    # ── Connection state ──

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_tick(self) -> Optional[TickEvent]:
        return self._last_tick

    def start(self) -> None:
        """Mark the feed as running (actual WS connect is async)."""
        self._running = True
        self._reconnect_count = 0
        logger.info(
            "Feed started: source=%s symbols=%s",
            self.config.source.value,
            self.config.symbols,
        )

    def stop(self) -> None:
        """Mark the feed as stopped."""
        self._running = False
        logger.info("Feed stopped")

    def handle_disconnect(self) -> bool:
        """Handle a disconnect; return True if should reconnect."""
        self._reconnect_count += 1
        if self._reconnect_count <= self.config.max_reconnects:
            logger.warning(
                "Reconnecting (%d/%d)...",
                self._reconnect_count,
                self.config.max_reconnects,
            )
            return True
        logger.error("Max reconnects reached — giving up")
        self._running = False
        return False
