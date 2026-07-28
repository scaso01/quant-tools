"""CLI entry point for the stock scanner."""

from __future__ import annotations

import argparse
import sys

from quant_tools.data.universe import SECTOR_MAP, get_sector_tickers, get_universe
from quant_tools.models import AssetClass, SignalType
from quant_tools.scanner.engine import scan_multi_asset, scan_universe


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the scanner CLI."""
    parser = argparse.ArgumentParser(
        prog="quant-tools scan",
        description="Scan stocks for trading signals.",
    )
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Universe to scan (default: sp500)",
    )
    parser.add_argument(
        "--sector",
        default=None,
        help=f"Filter by sector. Available: {', '.join(sorted(SECTOR_MAP.keys()))}",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        choices=[s.value for s in SignalType],
        help="Signal types to compute (default: all)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top results to show (default: 20)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Price history period (default: 1y)",
    )
    parser.add_argument(
        "--asset-class",
        default="equity",
        choices=["equity", "crypto", "forex", "commodity", "all"],
        help="Asset class to scan (default: equity). 'all' scans all asset classes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scanner CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve signals
    signal_types: list[SignalType] | None = None
    if args.signals:
        signal_types = [SignalType(s) for s in args.signals]

    ac_choice = args.asset_class

    # Multi-asset "all" mode
    if ac_choice == "all":
        assets: list[tuple[str, AssetClass]] = []
        # Add equity universe
        try:
            eq_symbols = get_universe(args.universe)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        assets.extend((s, AssetClass.EQUITY) for s in eq_symbols)
        # Add non-equity universes
        for ac_name in ("crypto", "forex", "commodity"):
            ac_symbols = get_universe(asset_class=ac_name)
            ac_enum = AssetClass(ac_name)
            assets.extend((s, ac_enum) for s in ac_symbols)

        print(f"Scanning {len(assets)} symbols across all asset classes...")
        results = scan_multi_asset(
            assets=assets,
            signals=signal_types,
            top_n=args.top_n,
            period=args.period,
        )
    else:
        # Single asset class
        asset_class_enum: AssetClass | None = None
        if ac_choice != "equity":
            asset_class_enum = AssetClass(ac_choice)
            try:
                symbols = get_universe(asset_class=ac_choice)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        elif args.sector:
            try:
                symbols = get_sector_tickers(args.sector)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        else:
            try:
                symbols = get_universe(args.universe)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

        print(f"Scanning {len(symbols)} {ac_choice} symbols...")
        results = scan_universe(
            symbols=symbols,
            signals=signal_types,
            top_n=args.top_n,
            period=args.period,
            asset_class=asset_class_enum,
        )

    if not results:
        print("No results found.")
        return 0

    # Print results table
    print(f"\n{'Rank':<5} {'Symbol':<8} {'Score':>6} {'Signal':<16} "
          f"{'Price':>9} {'1D%':>7} {'1M%':>7} {'RSI':>6}")
    print("-" * 72)

    for i, r in enumerate(results, 1):
        rsi_str = f"{r.rsi_14:.0f}" if r.rsi_14 is not None else "N/A"
        print(f"{i:<5} {r.symbol:<8} {r.score:>6.1f} {r.signal.value:<16} "
              f"{r.price:>9.2f} {r.change_1d:>+6.1f}% {r.change_1m:>+6.1f}% {rsi_str:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
