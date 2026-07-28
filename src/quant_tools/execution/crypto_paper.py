"""Crypto paper trading engine — simulated exchange for strategy testing.

Pure in-memory engine with no external dependencies. Tracks balances,
fills orders at current market price (no slippage model yet), and
computes PnL.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data types
# =============================================================================


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class PaperOrderStatus(Enum):
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass
class PaperOrder:
    """A simulated order with fill details."""

    order_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    qty: float = 0.0
    price: float = 0.0
    status: PaperOrderStatus = PaperOrderStatus.FILLED
    filled_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def notional(self) -> float:
        """Total notional value of the order."""
        return self.qty * self.price


# =============================================================================
# Paper trading engine
# =============================================================================


class CryptoPaperEngine:
    """Simulated crypto exchange for paper trading.

    Supports multiple base assets against a single quote currency (USD).
    Orders fill immediately at the given price with no slippage.

    Parameters
    ----------
    initial_balance:
        Starting USD balance.
    """

    def __init__(self, initial_balance: float = 100_000.0) -> None:
        self._usd_balance: float = initial_balance
        self._initial_balance: float = initial_balance
        # symbol -> quantity held
        self._positions: dict[str, float] = {}
        # All executed orders
        self._orders: list[PaperOrder] = []
        # Last known prices per symbol
        self._prices: dict[str, float] = {}

    # ── Properties ──

    @property
    def usd_balance(self) -> float:
        return self._usd_balance

    @property
    def positions(self) -> dict[str, float]:
        """Current positions (symbol -> qty). Only non-zero positions."""
        return {s: q for s, q in self._positions.items() if q > 1e-12}

    @property
    def order_history(self) -> list[PaperOrder]:
        return list(self._orders)

    # ── Order execution ──

    def submit_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
    ) -> PaperOrder:
        """Execute a paper order at the given price.

        Parameters
        ----------
        symbol:
            Trading pair (e.g., "BTC/USD").
        side:
            Buy or sell.
        qty:
            Quantity of the base asset.
        price:
            Execution price per unit.

        Returns
        -------
        PaperOrder with fill details or rejection reason.
        """
        self._prices[symbol] = price

        if qty <= 0:
            return self._reject(symbol, side, qty, price, "Quantity must be positive")

        if price <= 0:
            return self._reject(symbol, side, qty, price, "Price must be positive")

        if side == Side.BUY:
            return self._execute_buy(symbol, qty, price)
        else:
            return self._execute_sell(symbol, qty, price)

    def _execute_buy(self, symbol: str, qty: float, price: float) -> PaperOrder:
        """Execute a buy order."""
        cost = qty * price
        if cost > self._usd_balance:
            return self._reject(
                symbol, Side.BUY, qty, price,
                f"Insufficient balance: need ${cost:.2f}, have ${self._usd_balance:.2f}",
            )

        self._usd_balance -= cost
        self._positions[symbol] = self._positions.get(symbol, 0.0) + qty

        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=Side.BUY,
            qty=qty,
            price=price,
            status=PaperOrderStatus.FILLED,
            filled_at=datetime.now(),
        )
        self._orders.append(order)
        logger.info("BUY %.6f %s @ %.2f (cost: %.2f)", qty, symbol, price, cost)
        return order

    def _execute_sell(self, symbol: str, qty: float, price: float) -> PaperOrder:
        """Execute a sell order."""
        held = self._positions.get(symbol, 0.0)
        if qty > held + 1e-12:
            return self._reject(
                symbol, Side.SELL, qty, price,
                f"Insufficient holdings: need {qty:.6f}, have {held:.6f}",
            )

        proceeds = qty * price
        self._usd_balance += proceeds
        self._positions[symbol] = held - qty

        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=Side.SELL,
            qty=qty,
            price=price,
            status=PaperOrderStatus.FILLED,
            filled_at=datetime.now(),
        )
        self._orders.append(order)
        logger.info("SELL %.6f %s @ %.2f (proceeds: %.2f)", qty, symbol, price, proceeds)
        return order

    def _reject(
        self, symbol: str, side: Side, qty: float, price: float, reason: str
    ) -> PaperOrder:
        """Create a rejected order."""
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            status=PaperOrderStatus.REJECTED,
            error=reason,
        )
        self._orders.append(order)
        logger.warning("Order rejected: %s", reason)
        return order

    # ── Portfolio metrics ──

    def update_price(self, symbol: str, price: float) -> None:
        """Update the latest known price for a symbol."""
        self._prices[symbol] = price

    def equity(self) -> float:
        """Total equity = USD balance + market value of all positions."""
        position_value = sum(
            qty * self._prices.get(symbol, 0.0)
            for symbol, qty in self._positions.items()
        )
        return self._usd_balance + position_value

    def unrealized_pnl(self) -> float:
        """Unrealized PnL = equity - initial balance."""
        return self.equity() - self._initial_balance

    def realized_pnl(self) -> float:
        """Realized PnL from closed trades (sells).

        Simplified: sum of sell proceeds minus sum of buy costs for
        completed round-trips. This is approximate; a proper FIFO/LIFO
        tracker would be more accurate.
        """
        buy_cost = sum(
            o.notional for o in self._orders
            if o.side == Side.BUY and o.status == PaperOrderStatus.FILLED
        )
        sell_proceeds = sum(
            o.notional for o in self._orders
            if o.side == Side.SELL and o.status == PaperOrderStatus.FILLED
        )
        # Subtract current position value to get only realized portion
        position_cost = sum(
            qty * self._prices.get(symbol, 0.0)
            for symbol, qty in self._positions.items()
        )
        return sell_proceeds - buy_cost + position_cost

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._usd_balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._prices.clear()
