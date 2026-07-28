"""Prediction outcome resolution against actual market data."""
from __future__ import annotations

import logging
from datetime import datetime

from quant_tools.cache import PredictionOutcomeRow, PredictionRow, _lock
from quant_tools.models import PredictionDirection, PredictionOutcome
from quant_tools.predict.tracker import PredictionTracker

log = logging.getLogger("quant_tools")


class PredictionResolver:
    """Resolves pending predictions against actual price outcomes."""

    def __init__(self, tracker: PredictionTracker, registry=None) -> None:
        """
        Args:
            tracker: PredictionTracker for data access.
            registry: ProviderRegistry for fetching actual prices (optional, lazy imported).
        """
        self._tracker = tracker
        self._registry = registry

    def _get_registry(self):
        """Lazy-load ProviderRegistry."""
        if self._registry is None:
            from quant_tools.data.providers.registry import ProviderRegistry
            self._registry = ProviderRegistry()
        return self._registry

    def resolve_prediction(self, prediction: dict) -> PredictionOutcome | None:
        """Resolve a single prediction against actual data.

        Args:
            prediction: Prediction dict from tracker.get_pending().

        Returns:
            PredictionOutcome if resolved, None if actual data unavailable.
        """
        try:
            actual_direction, actual_magnitude = self._calculate_actual(
                prediction["symbol"],
                prediction["asset_class"],
                prediction["timestamp"],
                prediction["target_date"],
                row_interval=prediction.get("interval", "1d"),
            )
        except Exception as exc:
            log.warning("Could not resolve prediction %d: %s", prediction["id"], exc)
            return None

        predicted_dir = prediction["prediction_direction"]
        correct = (predicted_dir == actual_direction.value)

        # Calculate error
        pred_mag = prediction.get("predicted_magnitude")
        error_pct = 0.0
        if pred_mag is not None and actual_magnitude is not None:
            error_pct = abs(pred_mag - actual_magnitude)

        outcome = PredictionOutcome(
            prediction_id=prediction["id"],
            actual_direction=actual_direction,
            actual_magnitude=actual_magnitude,
            correct=correct,
            error_pct=error_pct,
        )

        # Store outcome and mark prediction as resolved
        with _lock:
            session = self._tracker._cache._session()
            try:
                session.add(PredictionOutcomeRow(
                    prediction_id=prediction["id"],
                    actual_direction=actual_direction.value,
                    actual_magnitude=actual_magnitude,
                    correct=1 if correct else 0,
                    error_pct=error_pct,
                    resolved_at=datetime.now(),
                ))
                # Mark prediction as resolved
                pred_row = session.query(PredictionRow).filter(PredictionRow.id == prediction["id"]).first()
                if pred_row:
                    pred_row.resolved = 1
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        return outcome

    def resolve_all_pending(self) -> list[PredictionOutcome]:
        """Resolve all pending past-due predictions.

        Returns:
            List of resolved PredictionOutcome objects.
        """
        pending = self._tracker.get_pending()
        outcomes: list[PredictionOutcome] = []
        models_updated: set[tuple[str, str]] = set()

        for pred in pending:
            outcome = self.resolve_prediction(pred)
            if outcome is not None:
                outcomes.append(outcome)
                models_updated.add((pred["model_name"], pred["asset_class"]))

        # Update aggregate performance for all affected models
        for model_name, asset_class in models_updated:
            try:
                self._tracker.update_performance(model_name, asset_class)
            except Exception as exc:
                log.warning("Could not update performance for %s/%s: %s", model_name, asset_class, exc)

        log.info("Resolved %d/%d pending predictions", len(outcomes), len(pending))
        return outcomes

    def _calculate_actual(
        self,
        symbol: str,
        asset_class: str,
        prediction_timestamp,
        target_date: str,
        row_interval: str = "1d",
    ) -> tuple[PredictionDirection, float | None]:
        """Fetch actual price and determine direction/magnitude.

        Args:
            symbol: Ticker symbol.
            asset_class: Asset class string.
            prediction_timestamp: When prediction was made.
            target_date: Target date string.  Accepts both ``YYYY-MM-DD``
                (daily) and ``YYYY-MM-DD HH:MM:SS`` (intraday).
            row_interval: Bar interval stored in the prediction row (e.g.
                ``"1d"``, ``"1h"``).  Used to fetch intraday OHLCV when the
                prediction was made on a sub-daily interval.

        Returns:
            Tuple of (actual_direction, actual_magnitude).
        """
        registry = self._get_registry()

        # Parse target_date -- try datetime first, fall back to date-only
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        if isinstance(prediction_timestamp, str):
            pred_dt = datetime.fromisoformat(prediction_timestamp)
        else:
            pred_dt = prediction_timestamp

        # Calculate period to cover prediction window + buffer
        try:
            days_span = max((target_dt - pred_dt).days, 1)
            # Add buffer: at least 2x the horizon or 60 days minimum
            fetch_days = max(days_span * 2, 60)
            if fetch_days <= 30:
                period = "1mo"
            elif fetch_days <= 90:
                period = "3mo"
            elif fetch_days <= 180:
                period = "6mo"
            else:
                period = "1y"
        except (ValueError, TypeError):
            period = "3mo"

        # For intraday predictions, shorten the fetch period and request
        # the matching bar interval so that resolution has enough
        # granularity.
        is_intraday = row_interval != "1d"
        if is_intraday:
            # For sub-day intervals we only need a few days of data
            intraday_days = max(days_span * 2, 7)
            if intraday_days <= 7:
                period = "5d"
            elif intraday_days <= 30:
                period = "1mo"
            else:
                period = "3mo"
            df = registry.fetch_ohlcv(
                symbol, period=period, interval=row_interval,
            )
        else:
            df = registry.fetch_ohlcv(symbol, period=period)

        if df is None or df.empty:
            raise ValueError(f"No price data for {symbol}")

        # Get close prices
        close = df["Close"]

        if len(close) < 2:
            raise ValueError(f"Insufficient price data for {symbol}")

        # Try date-based lookup to match prediction window
        start_price = close.iloc[0]
        end_price = close.iloc[-1]

        try:
            # target_dt and pred_dt already parsed above; reuse them
            # Only use date-based lookup if dates fall within data range
            if pred_dt < target_dt:
                idx = close.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_convert(None)

                first_date = idx[0].to_pydatetime() if hasattr(idx[0], "to_pydatetime") else idx[0]
                last_date = idx[-1].to_pydatetime() if hasattr(idx[-1], "to_pydatetime") else idx[-1]

                # Only use searchsorted if at least one date is within data range
                if pred_dt <= last_date or target_dt >= first_date:
                    pred_i = max(0, min(idx.searchsorted(pred_dt), len(close) - 1))
                    target_i = max(0, min(idx.searchsorted(target_dt), len(close) - 1))

                    if pred_i < target_i:
                        start_price = close.iloc[pred_i]
                        end_price = close.iloc[target_i]
        except (ValueError, TypeError, IndexError):
            pass  # Keep first/last fallback

        if start_price <= 0:
            raise ValueError(f"Invalid start price for {symbol}")

        actual_return = (end_price - start_price) / start_price

        if actual_return > 0.001:
            direction = PredictionDirection.UP
        elif actual_return < -0.001:
            direction = PredictionDirection.DOWN
        else:
            direction = PredictionDirection.NEUTRAL

        return direction, float(actual_return)
