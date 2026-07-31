"""CLI entry point for factor scoring commands."""

from __future__ import annotations

import argparse
import sys

from quant_tools.factors.macro import detect_regime
from quant_tools.factors.rebalance import get_recommendations
from quant_tools.factors.scoring import calculate_factor_scores

# Default universe for quick testing
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "DIS",
    "BAC", "XOM", "PFE", "KO", "PEP",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for factor commands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="quant-tools factors",
        description="Factor scoring and macro regime analysis.",
    )
    parser.add_argument(
        "--universe",
        nargs="+",
        default=DEFAULT_UNIVERSE,
        help=(
            "A universe name (sp500, sp100, russell1000, international, "
            "emerging, global) or a space-separated list of tickers "
            "(default: top 20 S&P 500)."
        ),
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Show current macro regime from FRED data.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top picks to display (default: 10).",
    )
    return parser


def _resolve_universe(values: list[str]) -> list[str]:
    """Turn --universe into a ticker list.

    ``scan`` takes a universe name here and the README documents the same shape
    for ``factors``, but this only ever accepted tickers. So the documented
    ``factors --universe sp100`` asked Yahoo for a stock called SP100, retried
    the 404 three times, and printed a single row of zeroes for it.

    A lone argument is tried as a universe name first and falls back to being a
    ticker, so ``--universe AAPL MSFT`` keeps working unchanged.
    """
    from quant_tools.data.universe import get_universe

    if len(values) == 1:
        try:
            return get_universe(values[0])
        except ValueError:
            pass  # Not a universe name, so treat it as a single ticker.
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the factor scoring CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.regime:
            print("Detecting macro regime from FRED data...\n")
            snapshot = detect_regime()
            print(f"Regime: {snapshot.regime.value.upper()}")
            print(f"GDP Growth: {snapshot.gdp_growth or 'N/A'}%")
            print(f"Unemployment: {snapshot.unemployment or 'N/A'}%")
            print(f"CPI YoY: {snapshot.cpi_yoy or 'N/A'}%")
            print(f"Fed Funds Rate: {snapshot.fed_funds_rate or 'N/A'}%")
            print(f"Yield Spread (10Y-2Y): {snapshot.yield_spread_10y2y or 'N/A'}%")
            print("\nSignals:")
            for key, val in snapshot.signals.items():
                print(f"  {key}: {val}")
            print()

        symbols = _resolve_universe(args.universe)
        print(f"Computing factor scores for {len(symbols)} symbols...")
        scores = calculate_factor_scores(symbols)

        print(f"\n{'Rank':<6}{'Symbol':<8}{'Name':<25}{'Value':<9}{'Mom':<9}{'Qual':<9}{'Composite':<10}")
        print("-" * 76)
        for fs in scores[:args.top_n]:
            print(
                f"{fs.rank:<6}{fs.symbol:<8}{fs.name[:24]:<25}"
                f"{fs.value_z:>+7.2f}  {fs.momentum_z:>+7.2f}  "
                f"{fs.quality_z:>+7.2f}  {fs.composite_z:>+8.3f}"
            )

        if args.regime:
            snapshot = detect_regime()
            rec = get_recommendations(snapshot.regime, scores, top_n=args.top_n)
            print(f"\n--- Regime Tilt: {rec.regime.value.upper()} ---")
            print(f"Value: {rec.value_tilt:.0%}  Momentum: {rec.momentum_tilt:.0%}  "
                  f"Quality: {rec.quality_tilt:.0%}  Defensive: {rec.defensive_tilt:.0%}")
            print(f"\n{rec.rationale}")
            print(f"\nTop {args.top_n} regime-tilted picks:")
            for fs in rec.top_picks:
                print(f"  {fs.rank}. {fs.symbol} ({fs.name[:20]}) composite={fs.composite_z:+.3f}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
