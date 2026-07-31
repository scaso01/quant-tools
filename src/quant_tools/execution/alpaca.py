"""Alpaca paper trading — submit orders via Alpaca API.

Wraps alpaca-py with circuit-breaker protection and graceful import-error
handling. All API calls are mockable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Try importing alpaca SDK — gracefully degrade if not installed
# =============================================================================

# alpaca-py, not the retired alpaca-trade-api. The old SDK pinned
# websockets<11 while Streamlit needs >=12, so the two could never be
# installed together and the [realtime] extra always failed to resolve.
_ALPACA_AVAILABLE = False
try:
    from alpaca.trading.client import TradingClient  # type: ignore[import-untyped]
    from alpaca.trading.requests import (  # type: ignore[import-untyped]
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
    )

    _ALPACA_AVAILABLE = True
except ImportError:
    TradingClient = None  # type: ignore[assignment,misc]
    LimitOrderRequest = MarketOrderRequest = None  # type: ignore[assignment,misc]
    StopLimitOrderRequest = StopOrderRequest = None  # type: ignore[assignment,misc]


# =============================================================================
# Data types
# =============================================================================


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of an order submission."""

    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    qty: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """A current position."""

    symbol: str = ""
    qty: float = 0.0
    avg_entry: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = 0.0


@dataclass
class AccountInfo:
    """Account summary."""

    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0


# =============================================================================
# Circuit breaker
# =============================================================================


class CircuitBreaker:
    """Simple circuit breaker to protect against API failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """True if circuit is open (blocking requests)."""
        if self._is_open:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                self._is_open = False
                self._failure_count = 0
                logger.info("Circuit breaker reset after %.0fs", elapsed)
                return False
        return self._is_open

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call — may trip the breaker."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._is_open = True
            logger.warning(
                "Circuit breaker OPEN after %d failures", self._failure_count
            )


# =============================================================================
# Alpaca paper trader
# =============================================================================


class AlpacaPaperTrader:
    """Paper trading via Alpaca API with circuit breaker.

    Parameters
    ----------
    api_key:
        Alpaca API key.
    api_secret:
        Alpaca API secret.
    base_url:
        Alpaca paper trading URL.
    api:
        Optional pre-built API client (for testing / injection).
    """

    PAPER_URL = "https://paper-api.alpaca.markets"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        api: Any = None,
    ) -> None:
        self._api = api
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url or self.PAPER_URL
        self._breaker = CircuitBreaker()

        if self._api is None and _ALPACA_AVAILABLE and api_key:
            # paper=True already points at paper-api.alpaca.markets, so only a
            # caller-supplied base_url needs url_override.
            override = base_url or None
            self._api = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=True,
                url_override=override,
            )

    @property
    def is_available(self) -> bool:
        """Whether the Alpaca SDK is installed and API is configured."""
        return self._api is not None

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    # ── Orders ──

    @staticmethod
    def _build_request(order_type: OrderType, kwargs: dict[str, Any]) -> Any:
        """Wrap order fields in the request object alpaca-py expects.

        Returns the plain dict when the SDK is absent so an injected test
        double still sees the fields it asserts on.
        """
        if not _ALPACA_AVAILABLE:
            return kwargs
        request_cls = {
            OrderType.MARKET: MarketOrderRequest,
            OrderType.LIMIT: LimitOrderRequest,
            OrderType.STOP: StopOrderRequest,
            OrderType.STOP_LIMIT: StopLimitOrderRequest,
        }[order_type]
        return request_cls(**kwargs)

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide = OrderSide.BUY,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit an order to Alpaca paper trading.

        Returns OrderResult with status and any error.
        """
        if not self.is_available:
            return OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                status=OrderStatus.FAILED,
                error="Alpaca SDK not available or not configured",
            )

        if self._breaker.is_open:
            return OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                status=OrderStatus.FAILED,
                error="Circuit breaker is open — API calls blocked",
            )

        try:
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "qty": qty,
                "side": side.value,
                "time_in_force": "day",
            }
            if limit_price is not None:
                kwargs["limit_price"] = limit_price
            if stop_price is not None:
                kwargs["stop_price"] = stop_price

            order = self._api.submit_order(
                order_data=self._build_request(order_type, kwargs)
            )
            self._breaker.record_success()

            return OrderResult(
                order_id=str(getattr(order, "id", "")),
                symbol=symbol,
                side=side,
                qty=qty,
                status=OrderStatus.FILLED,
                filled_price=getattr(order, "filled_avg_price", None),
                raw={"client_order_id": getattr(order, "client_order_id", "")},
            )
        except Exception as exc:
            self._breaker.record_failure()
            logger.error("Order submission failed: %s", exc)
            return OrderResult(
                symbol=symbol,
                side=side,
                qty=qty,
                status=OrderStatus.FAILED,
                error=str(exc),
            )

    # ── Positions ──

    def get_positions(self) -> list[Position]:
        """Fetch current positions."""
        if not self.is_available:
            return []

        if self._breaker.is_open:
            logger.warning("Circuit breaker open — cannot fetch positions")
            return []

        try:
            raw_positions = self._api.get_all_positions()
            self._breaker.record_success()
            positions = []
            for p in raw_positions:
                positions.append(
                    Position(
                        symbol=str(getattr(p, "symbol", "")),
                        qty=float(getattr(p, "qty", 0)),
                        avg_entry=float(getattr(p, "avg_entry_price", 0)),
                        current_price=float(getattr(p, "current_price", 0)),
                        unrealized_pnl=float(getattr(p, "unrealized_pl", 0)),
                        market_value=float(getattr(p, "market_value", 0)),
                    )
                )
            return positions
        except Exception as exc:
            self._breaker.record_failure()
            logger.error("Failed to fetch positions: %s", exc)
            return []

    # ── Account ──

    def get_account(self) -> Optional[AccountInfo]:
        """Fetch account information."""
        if not self.is_available:
            return None

        if self._breaker.is_open:
            logger.warning("Circuit breaker open — cannot fetch account")
            return None

        try:
            acct = self._api.get_account()
            self._breaker.record_success()
            return AccountInfo(
                equity=float(getattr(acct, "equity", 0)),
                cash=float(getattr(acct, "cash", 0)),
                buying_power=float(getattr(acct, "buying_power", 0)),
                portfolio_value=float(getattr(acct, "portfolio_value", 0)),
            )
        except Exception as exc:
            self._breaker.record_failure()
            logger.error("Failed to fetch account: %s", exc)
            return None
