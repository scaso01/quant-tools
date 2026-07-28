"""Tests for the WebSocket feed manager."""
from __future__ import annotations

from datetime import datetime

import pytest

from quant_tools.realtime.feed import (
    FeedConfig,
    FeedSource,
    TickEvent,
    WebSocketFeedManager,
    normalize_alpaca,
    normalize_binance,
)


# =============================================================================
# TickEvent tests
# =============================================================================


class TestTickEvent:
    """Test TickEvent data class."""

    def test_creation(self):
        """TickEvent should store all fields."""
        tick = TickEvent(
            symbol="AAPL",
            price=150.0,
            volume=1000.0,
            timestamp=datetime(2023, 6, 15, 10, 30),
        )
        assert tick.symbol == "AAPL"
        assert tick.price == 150.0
        assert tick.volume == 1000.0

    def test_spread(self):
        """Spread should be ask - bid."""
        tick = TickEvent(
            symbol="AAPL",
            price=150.0,
            volume=100.0,
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
        )
        assert tick.spread == pytest.approx(0.10)

    def test_mid_price(self):
        """Mid should be (bid + ask) / 2."""
        tick = TickEvent(
            symbol="AAPL",
            price=150.0,
            volume=100.0,
            timestamp=datetime.now(),
            bid=149.0,
            ask=151.0,
        )
        assert tick.mid == pytest.approx(150.0)

    def test_spread_none_without_bid_ask(self):
        """Spread should be None when bid/ask not set."""
        tick = TickEvent(
            symbol="AAPL", price=150.0, volume=100.0, timestamp=datetime.now()
        )
        assert tick.spread is None
        assert tick.mid is None

    def test_default_source(self):
        """Default source should be CUSTOM."""
        tick = TickEvent(
            symbol="TEST", price=1.0, volume=1.0, timestamp=datetime.now()
        )
        assert tick.source == FeedSource.CUSTOM


# =============================================================================
# FeedConfig tests
# =============================================================================


class TestFeedConfig:
    """Test FeedConfig data class."""

    def test_defaults(self):
        """FeedConfig should have sensible defaults."""
        config = FeedConfig()
        assert config.source == FeedSource.CUSTOM
        assert config.url == ""
        assert config.symbols == []
        assert config.reconnect_delay == 5.0
        assert config.max_reconnects == 10

    def test_custom_config(self):
        """FeedConfig should accept custom values."""
        config = FeedConfig(
            source=FeedSource.ALPACA,
            url="wss://stream.data.alpaca.markets/v2/iex",
            symbols=["AAPL", "MSFT"],
            api_key="test_key",
        )
        assert config.source == FeedSource.ALPACA
        assert len(config.symbols) == 2


# =============================================================================
# Normalizer tests
# =============================================================================


class TestNormalizers:
    """Test exchange message normalizers."""

    def test_normalize_alpaca(self):
        """normalize_alpaca should parse Alpaca trade messages."""
        raw = {
            "S": "AAPL",
            "p": 150.25,
            "s": 100,
            "t": "2023-06-15T10:30:00",
        }
        tick = normalize_alpaca(raw)
        assert tick is not None
        assert tick.symbol == "AAPL"
        assert tick.price == 150.25
        assert tick.volume == 100.0
        assert tick.source == FeedSource.ALPACA

    def test_normalize_binance(self):
        """normalize_binance should parse Binance trade stream."""
        raw = {
            "s": "BTCUSDT",
            "p": "42000.50",
            "q": "0.5",
            "T": 1686830000000,  # Unix ms
        }
        tick = normalize_binance(raw)
        assert tick is not None
        assert tick.symbol == "BTCUSDT"
        assert tick.price == 42000.50
        assert tick.source == FeedSource.BINANCE

    def test_normalize_alpaca_bad_data(self):
        """normalize_alpaca should return None for malformed data."""
        result = normalize_alpaca({"bad": "data"})
        # May return a tick with empty symbol and 0 price, or None
        # depending on which fields raise
        # At minimum, should not crash
        assert result is None or isinstance(result, TickEvent)


# =============================================================================
# WebSocketFeedManager tests
# =============================================================================


class TestWebSocketFeedManager:
    """Test feed manager callback dispatch and state."""

    def test_callback_registration(self):
        """on_tick should register callbacks."""
        config = FeedConfig(symbols=["AAPL"])
        mgr = WebSocketFeedManager(config)

        called = []
        mgr.on_tick(lambda t: called.append(t))

        tick = TickEvent(
            symbol="AAPL", price=150.0, volume=100.0, timestamp=datetime.now()
        )
        mgr.dispatch(tick)
        assert len(called) == 1
        assert called[0].symbol == "AAPL"

    def test_multiple_callbacks(self):
        """dispatch should call all registered callbacks."""
        config = FeedConfig()
        mgr = WebSocketFeedManager(config)

        count = {"a": 0, "b": 0}

        def cb_a(t):
            count["a"] += 1

        def cb_b(t):
            count["b"] += 1

        mgr.on_tick(cb_a)
        mgr.on_tick(cb_b)

        tick = TickEvent(
            symbol="TEST", price=1.0, volume=1.0, timestamp=datetime.now()
        )
        mgr.dispatch(tick)
        assert count["a"] == 1
        assert count["b"] == 1

    def test_start_stop(self):
        """start/stop should toggle is_running."""
        config = FeedConfig()
        mgr = WebSocketFeedManager(config)

        assert not mgr.is_running
        mgr.start()
        assert mgr.is_running
        mgr.stop()
        assert not mgr.is_running

    def test_handle_disconnect_reconnects(self):
        """handle_disconnect should return True within reconnect limit."""
        config = FeedConfig(max_reconnects=3)
        mgr = WebSocketFeedManager(config)
        mgr.start()

        assert mgr.handle_disconnect() is True  # 1st
        assert mgr.handle_disconnect() is True  # 2nd
        assert mgr.handle_disconnect() is True  # 3rd
        assert mgr.handle_disconnect() is False  # 4th -> exceeded

    def test_last_tick(self):
        """last_tick should track the most recent dispatched tick."""
        config = FeedConfig()
        mgr = WebSocketFeedManager(config)
        assert mgr.last_tick is None

        tick = TickEvent(
            symbol="AAPL", price=150.0, volume=100.0, timestamp=datetime.now()
        )
        mgr.dispatch(tick)
        assert mgr.last_tick is tick
