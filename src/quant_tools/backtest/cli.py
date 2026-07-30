"""CLI entry point for backtest commands."""

from __future__ import annotations

import argparse

import numpy as np

from quant_tools.backtest.engine import run_backtest
from quant_tools.backtest.strategies import (
    bollinger_strategy,
    momentum_strategy,
    rsi_strategy,
    sma_crossover,
)
from quant_tools.data.market_data import fetch_prices
from quant_tools.models import BacktestResult, StrategyType


_STRATEGY_MAP = {
    "sma_crossover": (StrategyType.SMA_CROSSOVER, sma_crossover),
    "bollinger": (StrategyType.BOLLINGER, bollinger_strategy),
    "rsi": (StrategyType.RSI, rsi_strategy),
    "momentum": (StrategyType.MOMENTUM, momentum_strategy),
}


def backtest_command(args: list[str] | None = None) -> BacktestResult:
    """Parse CLI args, fetch prices, run strategy and engine, print results.

    Args:
        args: Command-line argument list (defaults to sys.argv if None).

    Returns:
        BacktestResult from the engine.
    """
    parser = argparse.ArgumentParser(description="quant-tools backtest")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(_STRATEGY_MAP.keys()),
        help="Strategy to run",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA period (sma_crossover)")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA period (sma_crossover)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")

    parsed = parser.parse_args(args)

    # Fetch prices
    df = fetch_prices(parsed.ticker, start=parsed.start, end=parsed.end)
    close = df["Close"].values.astype(np.float64)

    # Run strategy
    strategy_type, strategy_fn = _STRATEGY_MAP[parsed.strategy]
    if parsed.strategy == "sma_crossover":
        positions = strategy_fn(close, fast=parsed.fast, slow=parsed.slow)
    else:
        positions = strategy_fn(close)

    params = {
        "strategy": parsed.strategy,
        "fast": parsed.fast,
        "slow": parsed.slow,
    }

    # Run engine
    result = run_backtest(
        close=close,
        positions=positions,
        initial_capital=parsed.capital,
        strategy=strategy_type,
        ticker=parsed.ticker.upper(),
        # The window the data actually covers, which is not necessarily the one
        # asked for: a request before the listing date starts at the listing.
        start_date=df.index[0].date(),
        end_date=df.index[-1].date(),
        params=params,
    )

    # Print results
    _print_result(result)
    return result


def _print_result(result: BacktestResult) -> None:
    """Print a formatted backtest result summary."""
    print(f"\n{'='*50}")
    print(f"  Backtest: {result.strategy.value} on {result.ticker}")
    if result.start_date and result.end_date:
        print(f"  {result.start_date} -> {result.end_date}")
    else:
        print("  (date range not recorded)")
    print(f"{'='*50}")
    print(f"  Initial Capital:  ${result.initial_capital:>12,.2f}")
    print(f"  Final Value:      ${result.final_value:>12,.2f}")
    print(f"  Total Return:      {result.total_return:>11.2%}")
    print(f"  CAGR:              {result.cagr:>11.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>11.2f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:>11.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown:>11.2%}")
    print(f"  DD Duration:       {result.max_drawdown_duration_days:>8d} days")
    print(f"  Win Rate:          {result.win_rate:>11.2%}")
    print(f"  Total Trades:      {result.total_trades:>8d}")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "inf"
    print(f"  Profit Factor:     {pf_str:>11s}")
    print(f"{'='*50}\n")
