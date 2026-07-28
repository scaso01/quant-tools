"""Tests for the crypto paper trading engine."""
from __future__ import annotations

import pytest

from quant_tools.execution.crypto_paper import (
    CryptoPaperEngine,
    PaperOrder,
    PaperOrderStatus,
    Side,
)


# =============================================================================
# Buy order tests
# =============================================================================


class TestBuyOrders:
    """Test buying functionality."""

    def test_buy_deducts_balance(self):
        """Buying should reduce USD balance by notional value."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        order = engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=50_000.0)
        assert order.status == PaperOrderStatus.FILLED
        assert engine.usd_balance == pytest.approx(50_000.0)

    def test_buy_adds_position(self):
        """Buying should create a position for the symbol."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=2.0, price=30_000.0)
        assert "BTC/USD" in engine.positions
        assert engine.positions["BTC/USD"] == pytest.approx(2.0)

    def test_buy_multiple_adds(self):
        """Multiple buys should accumulate position."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("ETH/USD", Side.BUY, qty=5.0, price=2_000.0)
        engine.submit_order("ETH/USD", Side.BUY, qty=3.0, price=2_100.0)
        assert engine.positions["ETH/USD"] == pytest.approx(8.0)

    def test_buy_insufficient_balance(self):
        """Buy that exceeds balance should be rejected."""
        engine = CryptoPaperEngine(initial_balance=10_000.0)
        order = engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=50_000.0)
        assert order.status == PaperOrderStatus.REJECTED
        assert "Insufficient balance" in order.error
        # Balance unchanged
        assert engine.usd_balance == pytest.approx(10_000.0)


# =============================================================================
# Sell order tests
# =============================================================================


class TestSellOrders:
    """Test selling functionality."""

    def test_sell_adds_to_balance(self):
        """Selling should increase USD balance."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        engine.submit_order("BTC/USD", Side.SELL, qty=1.0, price=45_000.0)
        # Started 100k, bought at 40k, sold at 45k -> 105k
        assert engine.usd_balance == pytest.approx(105_000.0)

    def test_sell_reduces_position(self):
        """Selling should reduce position quantity."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=2.0, price=30_000.0)
        engine.submit_order("BTC/USD", Side.SELL, qty=1.0, price=35_000.0)
        assert engine.positions["BTC/USD"] == pytest.approx(1.0)

    def test_sell_insufficient_holdings(self):
        """Selling more than held should be rejected."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        order = engine.submit_order("BTC/USD", Side.SELL, qty=2.0, price=45_000.0)
        assert order.status == PaperOrderStatus.REJECTED
        assert "Insufficient holdings" in order.error

    def test_sell_without_position(self):
        """Selling without any position should be rejected."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        order = engine.submit_order("BTC/USD", Side.SELL, qty=1.0, price=40_000.0)
        assert order.status == PaperOrderStatus.REJECTED


# =============================================================================
# PnL and equity tests
# =============================================================================


class TestPnLAndEquity:
    """Test PnL and equity calculations."""

    def test_equity_cash_only(self):
        """Equity with no positions should equal USD balance."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        assert engine.equity() == pytest.approx(100_000.0)

    def test_equity_with_position(self):
        """Equity should include position market value."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        # USD = 60k, position = 1 BTC @ 40k
        assert engine.equity() == pytest.approx(100_000.0)

    def test_equity_after_price_change(self):
        """Equity should reflect price changes."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        engine.update_price("BTC/USD", 45_000.0)
        # USD = 60k, position = 1 BTC @ 45k = 105k
        assert engine.equity() == pytest.approx(105_000.0)

    def test_unrealized_pnl(self):
        """Unrealized PnL should be equity - initial balance."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        engine.update_price("BTC/USD", 45_000.0)
        assert engine.unrealized_pnl() == pytest.approx(5_000.0)


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_zero_quantity_rejected(self):
        """Zero quantity should be rejected."""
        engine = CryptoPaperEngine()
        order = engine.submit_order("BTC/USD", Side.BUY, qty=0, price=40_000.0)
        assert order.status == PaperOrderStatus.REJECTED

    def test_negative_price_rejected(self):
        """Negative price should be rejected."""
        engine = CryptoPaperEngine()
        order = engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=-100.0)
        assert order.status == PaperOrderStatus.REJECTED

    def test_order_history(self):
        """All orders should be tracked in order_history."""
        engine = CryptoPaperEngine(initial_balance=100_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=40_000.0)
        engine.submit_order("ETH/USD", Side.BUY, qty=5.0, price=2_000.0)
        assert len(engine.order_history) == 2

    def test_reset(self):
        """reset() should restore initial state."""
        engine = CryptoPaperEngine(initial_balance=50_000.0)
        engine.submit_order("BTC/USD", Side.BUY, qty=1.0, price=30_000.0)
        engine.reset()
        assert engine.usd_balance == pytest.approx(50_000.0)
        assert engine.positions == {}
        assert engine.order_history == []

    def test_paper_order_notional(self):
        """PaperOrder.notional should be qty * price."""
        order = PaperOrder(qty=2.5, price=40_000.0)
        assert order.notional == pytest.approx(100_000.0)
