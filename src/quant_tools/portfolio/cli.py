"""CLI entry point for portfolio analysis."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from quant_tools.config import cfg
from quant_tools.errors import QuantToolsError
from quant_tools.portfolio.analyzer import analyze_portfolio
from quant_tools.portfolio.monte_carlo import run_monte_carlo


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for portfolio CLI."""
    parser = argparse.ArgumentParser(
        prog="quant-tools portfolio",
        description="Analyze a portfolio of holdings with Monte Carlo simulation.",
    )
    parser.add_argument(
        "--holdings",
        type=str,
        required=True,
        help='JSON string of holdings, e.g. \'{"AAPL":0.3,"BND":0.4,"GLD":0.3}\'',
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Look-back period for price data (default: 1y)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=cfg.monte_carlo_simulations,
        help=f"Number of Monte Carlo simulations (default: {cfg.monte_carlo_simulations})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=252,
        help="Monte Carlo time horizon in trading days (default: 252)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run portfolio analysis from CLI arguments.

    Args:
        argv: Command-line arguments. None uses sys.argv.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Parse holdings JSON
    try:
        holdings: dict[str, float] = json.loads(args.holdings)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON for --holdings: {exc}", file=sys.stderr)
        return 1

    if not holdings:
        print("Error: --holdings must contain at least one symbol", file=sys.stderr)
        return 1

    try:
        # Run portfolio analysis
        result = analyze_portfolio(holdings, period=args.period)

        print("=" * 60)
        print("PORTFOLIO ANALYSIS")
        print("=" * 60)
        print(f"  Total Return:     {result.total_return:>10.2%}")
        print(f"  Annualized Return:{result.annualized_return:>10.2%}")
        print(f"  Volatility:       {result.volatility:>10.2%}")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
        print(f"  Max Drawdown:     {result.max_drawdown:>10.2%}")

        print("\nHOLDINGS:")
        for h in result.holdings:
            print(f"  {h.symbol:<6} w={h.weight:.0%}  ret={h.total_return:.2%}  vol={h.volatility:.2%}  sharpe={h.sharpe:.2f}")

        if result.sector_exposure:
            print("\nSECTOR EXPOSURE:")
            for sector, weight in sorted(result.sector_exposure.items(), key=lambda x: -x[1]):
                print(f"  {sector:<30} {weight:.1%}")

        # Monte Carlo simulation using weighted portfolio returns
        weights = np.array([holdings[s] for s in holdings])
        # Reconstruct portfolio returns from holdings data
        # Use the analyze_portfolio result to get per-holding returns
        # For MC, we need the historical portfolio returns
        # Re-fetch and compute for MC input
        from quant_tools.data.market_data import fetch_prices

        returns_arrays = []
        for symbol in holdings:
            df = fetch_prices(symbol, period=args.period)
            close = df["Close"].values.astype(float)
            ret = np.diff(close) / close[:-1]
            returns_arrays.append(ret)

        min_len = min(len(r) for r in returns_arrays)
        returns_matrix = np.column_stack([r[:min_len] for r in returns_arrays])
        portfolio_returns = returns_matrix @ weights

        mc = run_monte_carlo(
            portfolio_returns,
            n_simulations=args.simulations,
            time_horizon_days=args.horizon,
        )

        print(f"\nMONTE CARLO ({mc.n_simulations} simulations, {mc.time_horizon_days} days)")
        print(f"  Mean Final Value:   ${mc.mean_final:>12,.0f}")
        print(f"  Median Final Value: ${mc.median_final:>12,.0f}")
        print(f"  5th Percentile:     ${mc.percentile_5:>12,.0f}")
        print(f"  95th Percentile:    ${mc.percentile_95:>12,.0f}")
        print(f"  Prob of Loss:       {mc.prob_loss:>10.1%}")
        print(f"  Prob of 10%+ Gain:  {mc.prob_gain_10pct:>10.1%}")

    except QuantToolsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
