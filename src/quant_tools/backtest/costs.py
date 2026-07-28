"""Transaction cost models for backtest simulations.

Provides three levels of cost modeling:
- ProportionalCosts: Fixed basis points per trade (Level 1).
- MarketImpactCosts: Square-root market impact model (Level 2).
- SpreadCosts: Bid-ask spread model (Level 3).
- apply_costs: Deduct costs from strategy returns.
"""

from __future__ import annotations

import abc

import numpy as np


class TransactionCostModel(abc.ABC):
    """Abstract base class for transaction cost models.

    Subclasses must implement calculate() which returns the cost per period
    as a fraction of portfolio value.
    """

    @abc.abstractmethod
    def calculate(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate transaction costs per period.

        Args:
            positions: Array of positions (-1, 0, 1) at each timestep.
            prices: Array of closing prices.
            volumes: Optional array of trading volumes.

        Returns:
            Array of costs per period as fraction of portfolio value.
            Same length as positions.
        """


class ProportionalCosts(TransactionCostModel):
    """Level 1: Fixed basis points per trade.

    Charges a fixed fraction on each position change.

    Args:
        bps: Cost in basis points per trade (default 10 bps = 0.10%).
    """

    def __init__(self, bps: float = 10.0) -> None:
        self.cost_fraction = bps / 10_000

    def calculate(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate proportional costs on position changes.

        Returns:
            Array of costs, same length as positions.
        """
        trades = np.abs(np.diff(positions, prepend=0))
        return trades * self.cost_fraction


class MarketImpactCosts(TransactionCostModel):
    """Level 2: Square-root market impact model.

    Impact = sigma * sqrt(Q / ADV)
    where Q = order size (in shares), ADV = average daily volume.

    This models the price impact of executing large orders relative to
    available liquidity.

    Args:
        sigma: Impact coefficient (default 0.01).
        adv_window: Window for computing average daily volume (default 20).
    """

    def __init__(self, sigma: float = 0.01, adv_window: int = 20) -> None:
        self.sigma = sigma
        self.adv_window = adv_window

    def calculate(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate market impact costs.

        If volumes is not provided, falls back to proportional cost estimate
        using sigma as the cost fraction.

        Returns:
            Array of costs, same length as positions.
        """
        n = len(positions)
        trades = np.abs(np.diff(positions, prepend=0))

        if volumes is None:
            # Fallback: treat sigma as proportional cost
            return trades * self.sigma

        costs = np.zeros(n, dtype=np.float64)
        # Compute rolling average daily volume
        for i in range(n):
            if trades[i] == 0:
                continue
            start = max(0, i - self.adv_window)
            adv = np.mean(volumes[start : i + 1]) if i > 0 else volumes[0]
            if adv < 1e-10:
                continue
            # Q is proportional to trade size (position change as fraction of volume)
            participation = trades[i]  # position unit change
            costs[i] = self.sigma * np.sqrt(participation / max(adv, 1e-10))

        return costs


class SpreadCosts(TransactionCostModel):
    """Level 3: Bid-ask spread model.

    Models the cost of crossing the spread on each trade. Entries happen
    at the ask price and exits at the bid price.

    Args:
        spread_bps: Half-spread in basis points (default 5 bps = 0.05%).
    """

    def __init__(self, spread_bps: float = 5.0) -> None:
        self.half_spread = spread_bps / 10_000

    def calculate(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate spread costs.

        Cost is charged as half-spread on each side of a round-trip trade.
        A position change incurs cost proportional to the absolute change.

        Returns:
            Array of costs, same length as positions.
        """
        trades = np.abs(np.diff(positions, prepend=0))
        # Full spread cost per trade = 2 * half_spread, but we model as
        # half_spread on each leg (entry and exit)
        return trades * self.half_spread


def apply_costs(
    returns: np.ndarray,
    positions: np.ndarray,
    cost_model: TransactionCostModel,
    prices: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
) -> np.ndarray:
    """Deduct transaction costs from strategy returns.

    Args:
        returns: Strategy returns array (before costs).
        positions: Position array.
        cost_model: A TransactionCostModel instance.
        prices: Optional price array (for impact models).
        volumes: Optional volume array (for impact models).

    Returns:
        Net returns array (returns minus costs).
    """
    effective_prices = prices if prices is not None else np.ones_like(positions)
    costs = cost_model.calculate(positions, effective_prices, volumes)
    # Costs and returns may have different lengths; align to returns
    cost_slice = costs[: len(returns)]
    return returns - cost_slice
