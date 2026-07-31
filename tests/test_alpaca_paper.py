"""Tests for the Alpaca paper trading module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quant_tools.execution.alpaca import (
    AlpacaPaperTrader,
    CircuitBreaker,
    OrderSide,
    OrderStatus,
    OrderType,
)


# =============================================================================
# CircuitBreaker tests
# =============================================================================


class TestCircuitBreaker:
    """Test circuit breaker logic."""

    def test_starts_closed(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker(failure_threshold=3)
        assert not cb.is_open

    def test_opens_after_threshold(self):
        """Breaker should open after failure_threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()  # 3rd -> opens
        assert cb.is_open

    def test_success_resets_count(self):
        """Successful call should reset failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # Only 1 after reset
        assert not cb.is_open

    def test_resets_after_timeout(self):
        """Breaker should auto-reset after timeout."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.0)
        cb.record_failure()
        # With timeout=0, it should immediately reset on next check
        assert not cb.is_open  # Timeout elapsed


# =============================================================================
# AlpacaPaperTrader tests — with mocked API
# =============================================================================


class TestAlpacaPaperTrader:
    """Test Alpaca paper trader with injected mock API."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock Alpaca API."""
        api = MagicMock()
        # submit_order returns mock order
        order = MagicMock()
        order.id = "order-123"
        order.filled_avg_price = 150.0
        order.client_order_id = "client-456"
        api.submit_order.return_value = order
        # list_positions returns mock positions
        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "10"
        pos.avg_entry_price = "145.0"
        pos.current_price = "150.0"
        pos.unrealized_pl = "50.0"
        pos.market_value = "1500.0"
        api.get_all_positions.return_value = [pos]
        # get_account returns mock account
        acct = MagicMock()
        acct.equity = "100000.0"
        acct.cash = "50000.0"
        acct.buying_power = "100000.0"
        acct.portfolio_value = "100000.0"
        api.get_account.return_value = acct
        return api

    @pytest.fixture
    def trader(self, mock_api):
        """Create trader with injected mock API."""
        return AlpacaPaperTrader(api=mock_api)

    def test_is_available_with_api(self, trader):
        """Trader with injected API should be available."""
        assert trader.is_available is True

    def test_is_not_available_without_api(self):
        """Trader without API should not be available."""
        t = AlpacaPaperTrader()
        assert t.is_available is False

    def test_submit_order_success(self, trader, mock_api):
        """Successful order should return FILLED status."""
        result = trader.submit_order("AAPL", 10, OrderSide.BUY)
        assert result.status == OrderStatus.FILLED
        assert result.order_id == "order-123"
        assert result.symbol == "AAPL"
        mock_api.submit_order.assert_called_once()

    def test_submit_order_without_api(self):
        """Order without API should return FAILED."""
        t = AlpacaPaperTrader()
        result = t.submit_order("AAPL", 10)
        assert result.status == OrderStatus.FAILED
        assert "not available" in result.error.lower()

    def test_submit_order_api_failure(self, trader, mock_api):
        """API exception should return FAILED and trip breaker."""
        mock_api.submit_order.side_effect = Exception("API error")
        result = trader.submit_order("AAPL", 10)
        assert result.status == OrderStatus.FAILED
        assert "API error" in result.error

    def test_submit_order_circuit_breaker(self, trader, mock_api):
        """Order should fail when circuit breaker is open."""
        # Trip the circuit breaker
        mock_api.submit_order.side_effect = Exception("fail")
        for _ in range(5):
            trader.submit_order("AAPL", 10)

        # Now even with a working API, should be blocked
        mock_api.submit_order.side_effect = None
        result = trader.submit_order("AAPL", 10)
        assert result.status == OrderStatus.FAILED
        assert "circuit breaker" in result.error.lower()

    def test_get_positions(self, trader):
        """get_positions should return Position objects."""
        positions = trader.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10.0

    def test_get_positions_without_api(self):
        """get_positions without API should return empty list."""
        t = AlpacaPaperTrader()
        assert t.get_positions() == []

    def test_get_account(self, trader):
        """get_account should return AccountInfo."""
        acct = trader.get_account()
        assert acct is not None
        assert acct.equity == 100000.0
        assert acct.cash == 50000.0

    def test_get_account_without_api(self):
        """get_account without API should return None."""
        t = AlpacaPaperTrader()
        assert t.get_account() is None

    def test_order_types(self, trader, mock_api):
        """Should support different order types."""
        result = trader.submit_order(
            "AAPL", 10, OrderSide.SELL, OrderType.LIMIT, limit_price=155.0
        )
        assert result.status == OrderStatus.FILLED
        sent = mock_api.submit_order.call_args.kwargs["order_data"]
        # With alpaca-py installed this is a LimitOrderRequest; without it the
        # fields pass straight through as a dict. Both shapes are valid, so the
        # assertion reads whichever one arrived rather than pinning the test to
        # whether the optional [realtime] extra happens to be present.
        limit_price = sent["limit_price"] if isinstance(sent, dict) else sent.limit_price
        side = sent["side"] if isinstance(sent, dict) else sent.side
        assert float(limit_price) == 155.0
        # The SDK coerces the plain string into its own OrderSide enum.
        assert getattr(side, "value", side) == "sell"

    def test_every_order_type_maps_to_a_request(self, trader, mock_api):
        """No OrderType may fall through _build_request with a KeyError.

        The request class is chosen by a lookup table, so a new OrderType added
        without a matching entry would raise instead of placing the order.
        """
        prices = {
            OrderType.LIMIT: {"limit_price": 155.0},
            OrderType.STOP: {"stop_price": 145.0},
            OrderType.STOP_LIMIT: {"limit_price": 155.0, "stop_price": 145.0},
        }
        for order_type in OrderType:
            result = trader.submit_order(
                "AAPL", 10, OrderSide.BUY, order_type, **prices.get(order_type, {})
            )
            assert result.status == OrderStatus.FILLED, (
                f"{order_type.value} order was rejected: {result.error}"
            )
