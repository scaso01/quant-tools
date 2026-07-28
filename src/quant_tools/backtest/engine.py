"""Vectorized backtest engine."""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray

from quant_tools.backtest.metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_sortino,
    calculate_total_return,
    calculate_win_rate,
)
from quant_tools.models import BacktestResult, StrategyType


def run_backtest(
    close: NDArray[np.float64],
    positions: NDArray[np.float64],
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    strategy: StrategyType = StrategyType.SMA_CROSSOVER,
    ticker: str = "UNKNOWN",
    start_date: date | None = None,
    end_date: date | None = None,
    params: dict | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Logic:
        1. Calculate daily returns from close prices.
        2. Apply positions to get strategy returns: position[i-1] * return[i].
        3. Deduct commission on position changes.
        4. Build equity curve from strategy returns.
        5. Calculate all metrics.
        6. Return BacktestResult.

    Args:
        close: Array of closing prices.
        positions: Array of positions (-1, 0, 1) from a strategy.
        initial_capital: Starting capital.
        commission: Commission per trade as a fraction (e.g., 0.001 = 0.1%).
        strategy: Strategy type enum for the result.
        ticker: Ticker symbol for the result.
        start_date: Backtest start date.
        end_date: Backtest end date.
        params: Strategy parameters dict.

    Returns:
        BacktestResult with all metrics populated.
    """
    n = len(close)

    if n < 2:
        return _empty_result(strategy, ticker, initial_capital, start_date, end_date, params)

    # Step 1: Daily returns from close prices
    daily_returns = np.diff(close) / close[:-1]  # length n-1

    # Step 2: Strategy returns — position[i-1] determines return[i]
    # positions is length n, daily_returns is length n-1
    # strategy_return[i] = positions[i] * daily_returns[i] for i in 0..n-2
    # (positions[i] is the position held at close of day i, earning return on day i+1)
    strategy_returns = positions[:-1] * daily_returns  # length n-1

    # Step 3: Deduct commission on position changes
    # position_changes[i] = positions[i+1] - positions[i], length n-1
    # Commission for a change at index i is charged against strategy_returns[i]
    position_changes = np.diff(positions)  # length n-1
    commission_costs = np.abs(position_changes) * commission  # length n-1
    strategy_returns = strategy_returns - commission_costs

    # Step 4: Build equity curve
    equity_curve = np.empty(n, dtype=np.float64)
    equity_curve[0] = initial_capital
    equity_curve[1:] = initial_capital * np.cumprod(1.0 + strategy_returns)

    # Step 5: Identify trades (position change != 0) and compute trade returns
    trade_returns = _extract_trade_returns(positions, daily_returns)

    # Step 6: Calculate all metrics
    total_ret = calculate_total_return(equity_curve)
    cagr = calculate_cagr(equity_curve)
    sharpe = calculate_sharpe(strategy_returns)
    sortino = calculate_sortino(strategy_returns)
    max_dd, max_dd_dur = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)

    return BacktestResult(
        strategy=strategy,
        ticker=ticker,
        start_date=start_date or date(2020, 1, 1),
        end_date=end_date or date(2020, 12, 31),
        initial_capital=initial_capital,
        final_value=float(equity_curve[-1]),
        total_return=total_ret,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        win_rate=win_rate,
        total_trades=len(trade_returns),
        profit_factor=profit_factor,
        equity_curve=equity_curve.tolist(),
        params=params or {},
    )


def _extract_trade_returns(
    positions: NDArray[np.float64], daily_returns: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Extract per-trade returns from position and return series.

    A trade is defined as a contiguous block of non-zero, same-sign positions.
    The trade return is the cumulative return over that block.
    """
    trade_returns: list[float] = []
    in_trade = False
    trade_cumulative = 0.0

    for i in range(len(daily_returns)):
        pos = positions[i]

        if pos != 0:
            if not in_trade:
                in_trade = True
                trade_cumulative = pos * daily_returns[i]
            else:
                trade_cumulative += pos * daily_returns[i]
        else:
            if in_trade:
                trade_returns.append(trade_cumulative)
                in_trade = False
                trade_cumulative = 0.0

    # Close any open trade
    if in_trade:
        trade_returns.append(trade_cumulative)

    return np.array(trade_returns, dtype=np.float64) if trade_returns else np.array([], dtype=np.float64)


def _empty_result(
    strategy: StrategyType,
    ticker: str,
    initial_capital: float,
    start_date: date | None,
    end_date: date | None,
    params: dict | None,
) -> BacktestResult:
    """Return a zeroed-out BacktestResult for degenerate inputs."""
    return BacktestResult(
        strategy=strategy,
        ticker=ticker,
        start_date=start_date or date(2020, 1, 1),
        end_date=end_date or date(2020, 12, 31),
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return=0.0,
        cagr=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_days=0,
        win_rate=0.0,
        total_trades=0,
        profit_factor=0.0,
        equity_curve=[initial_capital],
        params=params or {},
    )
