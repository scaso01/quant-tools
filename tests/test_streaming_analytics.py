"""Tests for the streaming analytics engine."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from quant_tools.realtime.analytics import (
    Alert,
    AlertConfig,
    AlertType,
    StreamingAnalytics,
)
from quant_tools.realtime.feed import FeedSource, TickEvent


# =============================================================================
# Helpers
# =============================================================================


def _make_tick(
    symbol: str = "AAPL",
    price: float = 150.0,
    volume: float = 100.0,
    ts: datetime | None = None,
) -> TickEvent:
    """Create a TickEvent with defaults."""
    return TickEvent(
        symbol=symbol,
        price=price,
        volume=volume,
        timestamp=ts or datetime.now(),
    )


# =============================================================================
# VWAP tests
# =============================================================================


class TestVWAP:
    """Test volume-weighted average price calculation."""

    def test_single_tick(self):
        """VWAP of a single tick = its price."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(price=100.0, volume=50.0))
        assert engine.vwap() == pytest.approx(100.0)

    def test_equal_volume(self):
        """Equal-volume ticks -> VWAP = simple average."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(price=100.0, volume=10.0))
        engine.on_tick(_make_tick(price=200.0, volume=10.0))
        assert engine.vwap() == pytest.approx(150.0)

    def test_weighted_volume(self):
        """Higher-volume tick should dominate VWAP."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(price=100.0, volume=90.0))
        engine.on_tick(_make_tick(price=200.0, volume=10.0))
        # VWAP = (100*90 + 200*10) / 100 = 110
        assert engine.vwap() == pytest.approx(110.0)

    def test_vwap_by_symbol(self):
        """VWAP filtered by symbol should only use matching ticks."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(symbol="AAPL", price=150.0, volume=100.0))
        engine.on_tick(_make_tick(symbol="MSFT", price=300.0, volume=100.0))
        assert engine.vwap("AAPL") == pytest.approx(150.0)
        assert engine.vwap("MSFT") == pytest.approx(300.0)

    def test_vwap_empty(self):
        """VWAP with no data should return 0."""
        engine = StreamingAnalytics()
        assert engine.vwap() == 0.0


# =============================================================================
# Rolling volatility tests
# =============================================================================


class TestRollingVolatility:
    """Test rolling volatility computation."""

    def test_zero_for_single_tick(self):
        """Single tick should return 0 (insufficient data)."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(price=100.0))
        assert engine.rolling_volatility() == 0.0

    def test_positive_for_varying_prices(self):
        """Varying prices should yield positive volatility."""
        engine = StreamingAnalytics()
        for p in [100, 102, 98, 105, 97, 103, 99, 101, 104, 96]:
            engine.on_tick(_make_tick(price=float(p)))
        vol = engine.rolling_volatility()
        assert vol > 0

    def test_constant_prices_zero_vol(self):
        """Constant prices should have zero or near-zero volatility."""
        engine = StreamingAnalytics()
        for _ in range(20):
            engine.on_tick(_make_tick(price=100.0))
        vol = engine.rolling_volatility()
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_volatility_by_symbol(self):
        """Volatility filtered by symbol."""
        engine = StreamingAnalytics()
        for p in [100, 110, 90, 105, 95]:
            engine.on_tick(_make_tick(symbol="AAPL", price=float(p)))
        for _ in range(5):
            engine.on_tick(_make_tick(symbol="MSFT", price=300.0))

        assert engine.rolling_volatility("AAPL") > 0
        assert engine.rolling_volatility("MSFT") == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Alert tests
# =============================================================================


class TestAlerts:
    """Test alert triggering."""

    def test_price_above_alert_fires(self):
        """PRICE_ABOVE alert should trigger when price exceeds threshold."""
        engine = StreamingAnalytics()
        alerts_fired: list[Alert] = []
        engine.on_alert(lambda a: alerts_fired.append(a))
        engine.add_alert(
            AlertConfig(
                alert_type=AlertType.PRICE_ABOVE,
                threshold=155.0,
                cooldown_seconds=0,
            )
        )

        engine.on_tick(_make_tick(price=150.0))  # Below — no alert
        assert len(alerts_fired) == 0

        engine.on_tick(_make_tick(price=160.0))  # Above — should fire
        assert len(alerts_fired) == 1
        assert alerts_fired[0].current_value == 160.0

    def test_price_below_alert_fires(self):
        """PRICE_BELOW alert should trigger when price drops below threshold."""
        engine = StreamingAnalytics()
        alerts_fired: list[Alert] = []
        engine.on_alert(lambda a: alerts_fired.append(a))
        engine.add_alert(
            AlertConfig(
                alert_type=AlertType.PRICE_BELOW,
                threshold=100.0,
                cooldown_seconds=0,
            )
        )

        engine.on_tick(_make_tick(price=110.0))
        assert len(alerts_fired) == 0

        engine.on_tick(_make_tick(price=95.0))
        assert len(alerts_fired) == 1

    def test_alert_cooldown(self):
        """Alert should not re-fire within cooldown period."""
        engine = StreamingAnalytics()
        alerts_fired: list[Alert] = []
        engine.on_alert(lambda a: alerts_fired.append(a))
        engine.add_alert(
            AlertConfig(
                alert_type=AlertType.PRICE_ABOVE,
                threshold=100.0,
                cooldown_seconds=60.0,
            )
        )

        now = datetime.now()
        engine.on_tick(_make_tick(price=110.0, ts=now))
        assert len(alerts_fired) == 1

        # Second tick within cooldown -> should NOT fire
        engine.on_tick(_make_tick(price=120.0, ts=now + timedelta(seconds=10)))
        assert len(alerts_fired) == 1

    def test_clear_alerts(self):
        """clear_alerts should remove all alerts."""
        engine = StreamingAnalytics()
        engine.add_alert(AlertConfig(alert_type=AlertType.PRICE_ABOVE, threshold=100))
        engine.add_alert(AlertConfig(alert_type=AlertType.PRICE_BELOW, threshold=50))
        engine.clear_alerts()

        alerts_fired: list[Alert] = []
        engine.on_alert(lambda a: alerts_fired.append(a))
        engine.on_tick(_make_tick(price=200.0))
        assert len(alerts_fired) == 0


# =============================================================================
# General engine tests
# =============================================================================


class TestStreamingAnalyticsGeneral:
    """Test general engine behavior."""

    def test_tick_count(self):
        """tick_count should track window size."""
        engine = StreamingAnalytics(window_size=5)
        for _ in range(3):
            engine.on_tick(_make_tick())
        assert engine.tick_count == 3

        # Fill past window size
        for _ in range(5):
            engine.on_tick(_make_tick())
        assert engine.tick_count == 5  # Capped at window_size

    def test_latest_price(self):
        """latest_price should return most recent tick price."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(price=100.0))
        engine.on_tick(_make_tick(price=200.0))
        assert engine.latest_price() == 200.0

    def test_total_volume(self):
        """total_volume should sum all tick volumes."""
        engine = StreamingAnalytics()
        engine.on_tick(_make_tick(volume=100.0))
        engine.on_tick(_make_tick(volume=200.0))
        assert engine.total_volume() == 300.0
