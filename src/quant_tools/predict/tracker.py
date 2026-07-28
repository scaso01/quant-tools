"""Prediction storage and performance tracking."""
from __future__ import annotations

import logging
from datetime import datetime

from quant_tools.cache import CacheManager, ModelPerformanceRow, PredictionOutcomeRow, PredictionRow, _lock
from quant_tools.models import (
    AssetClass,
    EnsemblePrediction,
    ModelPerformance,
    PredictionDirection,
    PredictionOutcome,
)

log = logging.getLogger("quant_tools")


class PredictionTracker:
    """Stores predictions and tracks model performance."""

    def __init__(self, cache_manager: CacheManager | None = None) -> None:
        if cache_manager is None:
            cache_manager = CacheManager()
        self._cache = cache_manager

    def store_prediction(self, prediction: EnsemblePrediction) -> int:
        """Store an ensemble prediction (individual models + ensemble result).

        Args:
            prediction: EnsemblePrediction with model_predictions list.

        Returns:
            Ensemble prediction ID.
        """
        with _lock:
            session = self._cache._session()
            try:
                # Store individual model predictions
                for mp in prediction.model_predictions:
                    row = PredictionRow(
                        timestamp=mp.timestamp,
                        symbol=prediction.symbol,
                        asset_class=prediction.asset_class.value,
                        model_name=mp.model_name,
                        prediction_direction=mp.direction.value,
                        predicted_magnitude=mp.magnitude,
                        confidence=mp.confidence,
                        horizon_days=prediction.horizon_days,
                        target_date=prediction.target_date,
                        interval=prediction.interval,
                        horizon_hours=mp.horizon_hours,
                        consensus_signal=prediction.consensus.value,
                        is_ensemble=0,
                        resolved=0,
                    )
                    session.add(row)

                # Store ensemble result
                ensemble_row = PredictionRow(
                    timestamp=prediction.timestamp,
                    symbol=prediction.symbol,
                    asset_class=prediction.asset_class.value,
                    model_name="ensemble",
                    prediction_direction=prediction.direction.value,
                    predicted_magnitude=prediction.magnitude,
                    confidence=prediction.confidence,
                    horizon_days=prediction.horizon_days,
                    target_date=prediction.target_date,
                    interval=prediction.interval,
                    horizon_hours=prediction.horizon_hours,
                    consensus_signal=prediction.consensus.value,
                    is_ensemble=1,
                    resolved=0,
                )
                session.add(ensemble_row)
                session.commit()

                return ensemble_row.id
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def get_pending(self) -> list[dict]:
        """Get all unresolved predictions where target_date <= now.

        Handles both daily (``YYYY-MM-DD``) and intraday
        (``YYYY-MM-DD HH:MM:SS``) target dates.  The comparison is
        lexicographic which works correctly for ISO-formatted strings
        when compared against the same-length reference timestamp.

        Returns:
            List of prediction dicts with all fields.
        """
        now_date = datetime.now().strftime("%Y-%m-%d")
        now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _lock:
            session = self._cache._session()
            try:
                # Fetch all unresolved rows (we filter in Python to handle
                # mixed date/datetime formats cleanly).
                rows = (
                    session.query(PredictionRow)
                    .filter(PredictionRow.resolved == 0)
                    .all()
                )
                results: list[dict] = []
                for r in rows:
                    td = r.target_date or ""
                    if len(td) <= 10:
                        # Daily format: compare against today's date
                        if td <= now_date:
                            results.append(self._row_to_dict(r))
                    else:
                        # Intraday datetime format: compare against now
                        if td <= now_datetime:
                            results.append(self._row_to_dict(r))
                return results
            finally:
                session.close()

    def get_latest_by_symbol(self, limit_per_symbol: int = 6) -> dict[str, list[dict]]:
        """Return the latest prediction set per symbol.

        Groups predictions by symbol, orders by timestamp desc, and returns
        the most recent ``limit_per_symbol`` rows for each symbol (typically
        1 ensemble + 5 individual model rows).

        Returns:
            Mapping of symbol -> list of prediction dicts (newest first).
        """
        from collections import defaultdict

        with _lock:
            session = self._cache._session()
            try:
                rows = (
                    session.query(PredictionRow)
                    .order_by(PredictionRow.timestamp.desc())
                    .all()
                )
                buckets: dict[str, list[dict]] = defaultdict(list)
                for r in rows:
                    sym = r.symbol
                    if len(buckets[sym]) < limit_per_symbol:
                        buckets[sym].append(self._row_to_dict(r))
                return dict(buckets)
            finally:
                session.close()

    def get_predictions(
        self,
        symbol: str | None = None,
        model: str | None = None,
        asset_class: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query predictions with optional filters."""
        with _lock:
            session = self._cache._session()
            try:
                query = session.query(PredictionRow)
                if symbol:
                    query = query.filter(PredictionRow.symbol == symbol)
                if model:
                    query = query.filter(PredictionRow.model_name == model)
                if asset_class:
                    query = query.filter(PredictionRow.asset_class == asset_class)
                rows = query.order_by(PredictionRow.timestamp.desc()).limit(limit).all()
                return [self._row_to_dict(r) for r in rows]
            finally:
                session.close()

    def get_performance(
        self,
        model_name: str | None = None,
        asset_class: str | None = None,
    ) -> list[ModelPerformance]:
        """Query model performance stats."""
        with _lock:
            session = self._cache._session()
            try:
                query = session.query(ModelPerformanceRow)
                if model_name:
                    query = query.filter(ModelPerformanceRow.model_name == model_name)
                if asset_class:
                    query = query.filter(ModelPerformanceRow.asset_class == asset_class)
                rows = query.all()
                return [
                    ModelPerformance(
                        model_name=r.model_name,
                        asset_class=AssetClass(r.asset_class),
                        total_predictions=r.total_predictions,
                        wins=r.wins,
                        losses=r.losses,
                        win_rate=r.win_rate or 0.0,
                        avg_confidence=r.avg_confidence or 0.0,
                        avg_error=r.avg_error or 0.0,
                        sharpe_of_predictions=r.sharpe_of_predictions or 0.0,
                    )
                    for r in rows
                ]
            finally:
                session.close()

    def update_performance(self, model_name: str, asset_class: str) -> None:
        """Recalculate aggregate stats for a model from resolved outcomes."""
        with _lock:
            session = self._cache._session()
            try:
                # Get all resolved predictions for this model+asset_class
                preds = (
                    session.query(PredictionRow)
                    .filter(
                        PredictionRow.model_name == model_name,
                        PredictionRow.asset_class == asset_class,
                        PredictionRow.resolved == 1,
                    )
                    .all()
                )
                pred_ids = [p.id for p in preds]

                if not pred_ids:
                    return

                outcomes = (
                    session.query(PredictionOutcomeRow)
                    .filter(PredictionOutcomeRow.prediction_id.in_(pred_ids))
                    .all()
                )

                if not outcomes:
                    return

                total = len(outcomes)
                wins = sum(1 for o in outcomes if o.correct)
                losses = total - wins
                win_rate = wins / total if total > 0 else 0.0
                avg_conf = sum(p.confidence or 0 for p in preds) / len(preds)
                errors = [o.error_pct for o in outcomes if o.error_pct is not None]
                avg_error = sum(errors) / len(errors) if errors else 0.0

                # Upsert performance row
                existing = (
                    session.query(ModelPerformanceRow)
                    .filter(
                        ModelPerformanceRow.model_name == model_name,
                        ModelPerformanceRow.asset_class == asset_class,
                    )
                    .first()
                )

                if existing:
                    existing.total_predictions = total
                    existing.wins = wins
                    existing.losses = losses
                    existing.win_rate = win_rate
                    existing.avg_confidence = avg_conf
                    existing.avg_error = avg_error
                    existing.last_updated = datetime.now()
                else:
                    session.add(ModelPerformanceRow(
                        model_name=model_name,
                        asset_class=asset_class,
                        total_predictions=total,
                        wins=wins,
                        losses=losses,
                        win_rate=win_rate,
                        avg_confidence=avg_conf,
                        avg_error=avg_error,
                        last_updated=datetime.now(),
                    ))

                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def clear_predictions(
        self,
        symbol: str | None = None,
        model: str | None = None,
        asset_class: str | None = None,
    ) -> int:
        """Delete predictions (and related outcomes/performance).

        Args:
            symbol: Delete only for this symbol. None = all.
            model: Delete only for this model. None = all.
            asset_class: Delete only for this asset class. None = all.

        Returns:
            Number of predictions deleted.
        """
        with _lock:
            session = self._cache._session()
            try:
                query = session.query(PredictionRow)
                if symbol:
                    query = query.filter(PredictionRow.symbol == symbol)
                if model:
                    query = query.filter(PredictionRow.model_name == model)
                if asset_class:
                    query = query.filter(PredictionRow.asset_class == asset_class)

                pred_ids = [r.id for r in query.all()]
                if not pred_ids:
                    return 0

                # Delete related outcomes
                session.query(PredictionOutcomeRow).filter(
                    PredictionOutcomeRow.prediction_id.in_(pred_ids)
                ).delete(synchronize_session=False)

                # Delete related performance rows
                if not symbol and not model and not asset_class:
                    session.query(ModelPerformanceRow).delete()
                else:
                    perf_filters = []
                    if model:
                        perf_filters.append(ModelPerformanceRow.model_name == model)
                    if asset_class:
                        perf_filters.append(ModelPerformanceRow.asset_class == asset_class)
                    if perf_filters:
                        session.query(ModelPerformanceRow).filter(*perf_filters).delete(
                            synchronize_session=False
                        )

                # Delete predictions
                count = query.delete(synchronize_session=False)
                session.commit()
                return count
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    @staticmethod
    def _row_to_dict(row: PredictionRow) -> dict:
        return {
            "id": row.id,
            "timestamp": row.timestamp,
            "symbol": row.symbol,
            "asset_class": row.asset_class,
            "model_name": row.model_name,
            "prediction_direction": row.prediction_direction,
            "predicted_magnitude": row.predicted_magnitude,
            "confidence": row.confidence,
            "horizon_days": row.horizon_days,
            "target_date": row.target_date,
            "interval": getattr(row, "interval", "1d") or "1d",
            "horizon_hours": getattr(row, "horizon_hours", None),
            "consensus_signal": row.consensus_signal,
            "is_ensemble": row.is_ensemble,
            "resolved": row.resolved,
        }
