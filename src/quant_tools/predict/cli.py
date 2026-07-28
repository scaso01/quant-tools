"""CLI commands for the prediction engine."""
from __future__ import annotations

import argparse
import sys


def build_predict_parser() -> argparse.ArgumentParser:
    """Build argument parser for predict command."""
    parser = argparse.ArgumentParser(
        prog="quant-tools predict",
        description="Generate ML ensemble predictions for any asset",
    )
    parser.add_argument("symbol", help="Symbol to predict (e.g., AAPL, BTC/USDT, EUR/USD)")
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument("--horizon", type=int, default=None, help="Prediction horizon in days (default: 5)")
    horizon_group.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Prediction horizon in hours (for intraday, mutually exclusive with --horizon)",
    )
    parser.add_argument(
        "--asset-class",
        choices=["equity", "crypto", "forex", "commodity"],
        default=None,
        help="Force asset class (auto-detected if omitted)",
    )
    return parser


def build_track_parser() -> argparse.ArgumentParser:
    """Build argument parser for track command."""
    parser = argparse.ArgumentParser(
        prog="quant-tools track",
        description="View stored predictions and model performance",
    )
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--asset-class", default=None, help="Filter by asset class")
    parser.add_argument("--limit", type=int, default=20, help="Max predictions to show")
    return parser


def build_resolve_parser() -> argparse.ArgumentParser:
    """Build argument parser for resolve command."""
    parser = argparse.ArgumentParser(
        prog="quant-tools resolve",
        description="Resolve pending predictions against actual outcomes",
    )
    return parser


def build_clear_parser() -> argparse.ArgumentParser:
    """Build argument parser for clear command."""
    parser = argparse.ArgumentParser(
        prog="quant-tools clear",
        description="Clear stored predictions and outcomes",
    )
    parser.add_argument("--symbol", default=None, help="Clear only for this symbol")
    parser.add_argument("--model", default=None, help="Clear only for this model")
    parser.add_argument("--asset-class", default=None, help="Clear only for this asset class")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    return parser


def cmd_predict(argv: list[str] | None = None) -> int:
    """Run ensemble prediction for a symbol."""
    parser = build_predict_parser()
    args = parser.parse_args(argv)

    try:
        from quant_tools.data.providers.registry import ProviderRegistry
        from quant_tools.models import AssetClass
        from quant_tools.predict.ensemble import PredictionEnsemble
        from quant_tools.predict.horizons import get_horizon_labels, interval_for_hours
        from quant_tools.predict.tracker import PredictionTracker

        registry = ProviderRegistry()

        # Auto-detect or use specified asset class
        if args.asset_class:
            asset_class = AssetClass(args.asset_class)
        else:
            asset_class = registry.detect_asset_class(args.symbol)

        # Determine intraday vs daily mode
        if args.horizon is None and args.hours is None:
            args.horizon = 5  # default when neither flag given
        use_hours = args.hours is not None
        if use_hours:
            # Validate against asset-class-aware presets (warn, don't error)
            valid_labels = get_horizon_labels(asset_class)
            chosen_label = f"{args.hours}h"
            if chosen_label not in valid_labels:
                print(
                    f"Warning: {chosen_label} is not a standard horizon for "
                    f"{asset_class.value}. Valid: {', '.join(valid_labels)}"
                )
            interval = interval_for_hours(args.hours)
            print(f"Fetching data for {args.symbol} ({asset_class.value})...")
            ohlcv = registry.fetch_ohlcv(args.symbol, period="60d", interval=interval)
        else:
            print(f"Fetching data for {args.symbol} ({asset_class.value})...")
            ohlcv = registry.fetch_ohlcv(args.symbol, period="2y")

        if ohlcv is None or ohlcv.empty:
            print(f"Error: No data available for {args.symbol}")
            return 1

        ensemble = PredictionEnsemble()
        if use_hours:
            print(f"Running ensemble prediction (horizon: {args.hours} hours)...")
            result = ensemble.predict(
                ohlcv, args.symbol, asset_class,
                horizon_days=None, hours=args.hours, interval=interval,
            )
        else:
            print(f"Running ensemble prediction (horizon: {args.horizon} days)...")
            result = ensemble.predict(ohlcv, args.symbol, asset_class, args.horizon)

        # Store prediction
        tracker = PredictionTracker()
        pred_id = tracker.store_prediction(result)

        # Display result
        print()
        print(f"{'=' * 60}")
        print(f"  PREDICTION: {args.symbol} ({asset_class.value})")
        print(f"{'=' * 60}")
        print(f"  Consensus:   {result.consensus.value.upper()}")
        print(f"  Direction:   {result.direction.value.upper()}")
        print(f"  Confidence:  {result.confidence:.1%}")
        if result.magnitude is not None:
            print(f"  Magnitude:   {result.magnitude:+.2%}")
        if use_hours:
            print(f"  Horizon:     {args.hours} hours")
            print(f"  Interval:    {interval}")
        else:
            print(f"  Horizon:     {result.horizon_days} days")
        print(f"  Target Date: {result.target_date}")
        print(f"  Stored as:   #{pred_id}")
        print()

        # Model breakdown
        print("  Model Breakdown:")
        print(f"  {'Model':<20} {'Direction':<10} {'Confidence':<12} {'Magnitude'}")
        print(f"  {'-' * 60}")
        for mp in result.model_predictions:
            mag_str = f"{mp.magnitude:+.2%}" if mp.magnitude is not None else "N/A"
            print(f"  {mp.model_name:<20} {mp.direction.value:<10} {mp.confidence:<12.1%} {mag_str}")
        print()

        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_track(argv: list[str] | None = None) -> int:
    """Show stored predictions and performance stats."""
    parser = build_track_parser()
    args = parser.parse_args(argv)

    try:
        from quant_tools.predict.tracker import PredictionTracker

        tracker = PredictionTracker()

        # Show predictions
        predictions = tracker.get_predictions(
            symbol=args.symbol,
            model=args.model,
            asset_class=args.asset_class,
            limit=args.limit,
        )

        if predictions:
            print(f"\n{'=' * 80}")
            print(f"  Recent Predictions ({len(predictions)} shown)")
            print(f"{'=' * 80}")
            print(f"  {'Symbol':<12} {'Class':<8} {'Model':<18} {'Dir':<8} {'Conf':<8} {'Target':<12} {'Status'}")
            print(f"  {'-' * 78}")
            for p in predictions:
                status = "Resolved" if p.get("resolved") else "Pending"
                conf = p.get("confidence", 0)
                print(
                    f"  {p['symbol']:<12} {p['asset_class']:<8} {p['model_name']:<18} "
                    f"{p['prediction_direction']:<8} {conf:<8.1%} {p['target_date']:<12} {status}"
                )
        else:
            print("\nNo predictions found.")

        # Show performance
        performance = tracker.get_performance()
        if performance:
            print(f"\n{'=' * 80}")
            print("  Model Performance")
            print(f"{'=' * 80}")
            print(f"  {'Model':<18} {'Asset':<10} {'Total':<8} {'Wins':<8} {'Win Rate':<10} {'Avg Conf'}")
            print(f"  {'-' * 70}")
            for p in performance:
                ac = p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class)
                print(
                    f"  {p.model_name:<18} {ac:<10} {p.total_predictions:<8} "
                    f"{p.wins:<8} {p.win_rate:<10.1%} {p.avg_confidence:.1%}"
                )
        print()

        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_clear(argv: list[str] | None = None) -> int:
    """Clear stored predictions."""
    parser = build_clear_parser()
    args = parser.parse_args(argv)

    try:
        from quant_tools.predict.tracker import PredictionTracker
        tracker = PredictionTracker()

        # Describe what will be deleted
        scope_parts = []
        if args.symbol:
            scope_parts.append(f"symbol={args.symbol}")
        if args.model:
            scope_parts.append(f"model={args.model}")
        if args.asset_class:
            scope_parts.append(f"asset_class={args.asset_class}")
        scope = ", ".join(scope_parts) if scope_parts else "ALL predictions"

        if not args.yes:
            confirm = input(f"Clear {scope}? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                return 0

        count = tracker.clear_predictions(
            symbol=args.symbol,
            model=args.model,
            asset_class=args.asset_class,
        )
        print(f"Cleared {count} predictions.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_resolve(argv: list[str] | None = None) -> int:
    """Resolve all pending past-due predictions."""
    try:
        from quant_tools.predict.resolver import PredictionResolver
        from quant_tools.predict.tracker import PredictionTracker

        tracker = PredictionTracker()
        resolver = PredictionResolver(tracker)

        print("Resolving pending predictions...")
        outcomes = resolver.resolve_all_pending()

        if outcomes:
            print(f"\nResolved {len(outcomes)} predictions:")
            for o in outcomes:
                status = "CORRECT" if o.correct else "WRONG"
                print(f"  #{o.prediction_id}: {o.actual_direction.value.upper()} — {status} (error: {o.error_pct:.2%})")
        else:
            print("No pending predictions to resolve.")

        print()
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
