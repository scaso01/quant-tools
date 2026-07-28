"""Streaming analytics — real-time VWAP, rolling volatility, and alerts.

Operates on a sliding window of TickEvents. All computations are pure
numpy/pandas — no network calls.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import numpy as np

from quant_tools.realtime.feed import TickEvent

logger = logging.getLogger(__name__)


# =============================================================================
# Alert system
# =============================================================================


class AlertType(Enum):
    """Types of streaming alerts."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_SPIKE = "volume_spike"


@dataclass
class AlertConfig:
    """Configuration for a single alert."""

    alert_type: AlertType
    threshold: float
    symbol: str = ""
    cooldown_seconds: float = 60.0
    message: str = ""


@dataclass
class Alert:
    """A triggered alert."""

    config: AlertConfig
    triggered_at: datetime = field(default_factory=datetime.now)
    current_value: float = 0.0
    message: str = ""


# =============================================================================
# Streaming analytics engine
# =============================================================================


AlertCallback = Callable[[Alert], None]


class StreamingAnalytics:
    """Real-time analytics engine operating on a sliding tick window.

    Tracks VWAP, rolling volatility, and fires alerts when thresholds
    are crossed.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._ticks: deque[TickEvent] = deque(maxlen=window_size)
        self._alerts: list[AlertConfig] = []
        self._alert_callbacks: list[AlertCallback] = []
        self._last_alert_times: dict[int, datetime] = {}

    # ── Tick ingestion ──

    def on_tick(self, tick: TickEvent) -> None:
        """Process a new tick — update window and check alerts."""
        self._ticks.append(tick)
        self._check_alerts(tick)

    @property
    def tick_count(self) -> int:
        """Number of ticks in the current window."""
        return len(self._ticks)

    # ── VWAP ──

    def vwap(self, symbol: Optional[str] = None) -> float:
        """Volume-weighted average price for ticks in the window.

        Parameters
        ----------
        symbol:
            Filter to a specific symbol. None = all ticks.

        Returns
        -------
        VWAP value, or 0.0 if no data.
        """
        ticks = self._filter_ticks(symbol)
        if not ticks:
            return 0.0

        prices = np.array([t.price for t in ticks])
        volumes = np.array([t.volume for t in ticks])

        total_volume = volumes.sum()
        if total_volume == 0:
            return float(prices.mean())

        return float(np.sum(prices * volumes) / total_volume)

    # ── Rolling volatility ──

    def rolling_volatility(
        self, symbol: Optional[str] = None, min_ticks: int = 2
    ) -> float:
        """Annualized volatility from tick-to-tick returns.

        Parameters
        ----------
        symbol:
            Filter to a specific symbol.
        min_ticks:
            Minimum ticks required (default 2).

        Returns
        -------
        Annualized volatility estimate, or 0.0 if insufficient data.
        """
        ticks = self._filter_ticks(symbol)
        if len(ticks) < min_ticks:
            return 0.0

        prices = np.array([t.price for t in ticks])
        if np.any(prices <= 0):
            return 0.0

        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            return 0.0

        # Annualize assuming ~6.5h trading * 252 days * ~1 tick/sec
        # Rough: use simple std * sqrt(N) where N approximates daily bars
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        return std * np.sqrt(252)

    # ── Latest price / volume ──

    def latest_price(self, symbol: Optional[str] = None) -> float:
        """Most recent price for a symbol."""
        ticks = self._filter_ticks(symbol)
        return ticks[-1].price if ticks else 0.0

    def total_volume(self, symbol: Optional[str] = None) -> float:
        """Total volume in the window."""
        ticks = self._filter_ticks(symbol)
        return sum(t.volume for t in ticks)

    # ── Alert management ──

    def add_alert(self, config: AlertConfig) -> None:
        """Register an alert configuration."""
        self._alerts.append(config)

    def on_alert(self, callback: AlertCallback) -> None:
        """Register a callback for triggered alerts."""
        self._alert_callbacks.append(callback)

    def clear_alerts(self) -> None:
        """Remove all alert configurations."""
        self._alerts.clear()
        self._last_alert_times.clear()

    def _check_alerts(self, tick: TickEvent) -> None:
        """Evaluate all alerts against the latest tick."""
        now = tick.timestamp
        for i, cfg in enumerate(self._alerts):
            # Symbol filter
            if cfg.symbol and cfg.symbol != tick.symbol:
                continue

            # Cooldown
            last = self._last_alert_times.get(i)
            if last is not None:
                elapsed = (now - last).total_seconds()
                if elapsed < cfg.cooldown_seconds:
                    continue

            triggered = False
            current_value = tick.price

            if cfg.alert_type == AlertType.PRICE_ABOVE:
                triggered = tick.price > cfg.threshold
            elif cfg.alert_type == AlertType.PRICE_BELOW:
                triggered = tick.price < cfg.threshold
            elif cfg.alert_type == AlertType.VOLATILITY_SPIKE:
                vol = self.rolling_volatility(tick.symbol)
                triggered = vol > cfg.threshold
                current_value = vol
            elif cfg.alert_type == AlertType.VOLUME_SPIKE:
                current_value = tick.volume
                triggered = tick.volume > cfg.threshold

            if triggered:
                self._last_alert_times[i] = now
                alert = Alert(
                    config=cfg,
                    triggered_at=now,
                    current_value=current_value,
                    message=cfg.message or f"{cfg.alert_type.value}: {current_value:.4f}",
                )
                for cb in self._alert_callbacks:
                    try:
                        cb(alert)
                    except Exception:
                        logger.exception("Alert callback failed")

    # ── Internal helpers ──

    def _filter_ticks(self, symbol: Optional[str]) -> list[TickEvent]:
        """Filter window ticks by symbol."""
        if symbol is None:
            return list(self._ticks)
        return [t for t in self._ticks if t.symbol == symbol]
