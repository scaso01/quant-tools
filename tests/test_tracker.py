"""Tests for PredictionTracker — storage, querying, and performance tracking."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from quant_tools.cache import CacheManager, ModelPerformanceRow, PredictionOutcomeRow, PredictionRow, _lock
from quant_tools.models import (
    AssetClass,
    ConsensusSignal,
    EnsemblePrediction,
    ModelPerformance,
    PredictionDirection,
    PredictionResult,
)
from quant_tools.predict.tracker import PredictionTracker


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_db):
    """Create a CacheManager with in-memory-style temp DB."""
    return CacheManager(db_path=tmp_db)


@pytest.fixture
def tracker(cache):
    """Create a PredictionTracker backed by temp DB."""
    return PredictionTracker(cache_manager=cache)


def _make_ensemble(
    symbol: str = "AAPL",
    asset_class: AssetClass = AssetClass.EQUITY,
    direction: PredictionDirection = PredictionDirection.UP,
    confidence: float = 0.8,
    magnitude: float = 0.05,
    horizon_days: int = 5,
    target_date: str = "2025-01-10",
    consensus: ConsensusSignal = ConsensusSignal.BUY,
    n_models: int = 3,
) -> EnsemblePrediction:
    """Helper to build an EnsemblePrediction with N model sub-predictions."""
    now = datetime.now()
    model_names = ["gradient_boost", "stats_forecast", "prophet", "garch", "neural"]
    preds = []
    for i in range(n_models):
        preds.append(PredictionResult(
            symbol=symbol,
            asset_class=asset_class,
            model_name=model_names[i % len(model_names)],
            direction=direction,
            magnitude=magnitude + i * 0.01,
            confidence=confidence - i * 0.05,
            horizon_days=horizon_days,
            target_date=target_date,
            timestamp=now,
        ))
    return EnsemblePrediction(
        symbol=symbol,
        asset_class=asset_class,
        consensus=consensus,
        confidence=confidence,
        direction=direction,
        magnitude=magnitude,
        horizon_days=horizon_days,
        target_date=target_date,
        model_predictions=preds,
        timestamp=now,
    )


# ── store_prediction tests ──────────────────────────────────────────────


class TestStorePrediction:
    def test_stores_ensemble_row(self, tracker, cache):
        ep = _make_ensemble(n_models=2)
        eid = tracker.store_prediction(ep)
        assert eid is not None and eid > 0

        session = cache._session()
        try:
            ensemble = session.query(PredictionRow).filter(PredictionRow.id == eid).first()
            assert ensemble is not None
            assert ensemble.is_ensemble == 1
            assert ensemble.model_name == "ensemble"
        finally:
            session.close()

    def test_stores_individual_model_rows(self, tracker, cache):
        ep = _make_ensemble(n_models=3)
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            models = session.query(PredictionRow).filter(PredictionRow.is_ensemble == 0).all()
            assert len(models) == 3
            names = {m.model_name for m in models}
            assert "gradient_boost" in names
            assert "stats_forecast" in names
            assert "prophet" in names
        finally:
            session.close()

    def test_total_rows_count(self, tracker, cache):
        ep = _make_ensemble(n_models=4)
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            total = session.query(PredictionRow).count()
            assert total == 5  # 4 models + 1 ensemble
        finally:
            session.close()

    def test_stores_correct_symbol(self, tracker, cache):
        ep = _make_ensemble(symbol="BTC-USD")
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            rows = session.query(PredictionRow).all()
            assert all(r.symbol == "BTC-USD" for r in rows)
        finally:
            session.close()

    def test_stores_correct_asset_class(self, tracker, cache):
        ep = _make_ensemble(asset_class=AssetClass.CRYPTO)
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            rows = session.query(PredictionRow).all()
            assert all(r.asset_class == "crypto" for r in rows)
        finally:
            session.close()

    def test_stores_direction(self, tracker, cache):
        ep = _make_ensemble(direction=PredictionDirection.DOWN)
        eid = tracker.store_prediction(ep)

        session = cache._session()
        try:
            ensemble = session.query(PredictionRow).filter(PredictionRow.id == eid).first()
            assert ensemble.prediction_direction == "down"
        finally:
            session.close()

    def test_stores_confidence(self, tracker, cache):
        ep = _make_ensemble(confidence=0.92)
        eid = tracker.store_prediction(ep)

        session = cache._session()
        try:
            ensemble = session.query(PredictionRow).filter(PredictionRow.id == eid).first()
            assert abs(ensemble.confidence - 0.92) < 1e-6
        finally:
            session.close()

    def test_stores_target_date(self, tracker, cache):
        ep = _make_ensemble(target_date="2026-06-15")
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            row = session.query(PredictionRow).filter(PredictionRow.is_ensemble == 1).first()
            assert row.target_date == "2026-06-15"
        finally:
            session.close()

    def test_stores_consensus_signal(self, tracker, cache):
        ep = _make_ensemble(consensus=ConsensusSignal.STRONG_SELL)
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            row = session.query(PredictionRow).filter(PredictionRow.is_ensemble == 1).first()
            assert row.consensus_signal == "strong_sell"
        finally:
            session.close()

    def test_new_predictions_unresolved(self, tracker, cache):
        ep = _make_ensemble()
        tracker.store_prediction(ep)

        session = cache._session()
        try:
            rows = session.query(PredictionRow).all()
            assert all(r.resolved == 0 for r in rows)
        finally:
            session.close()

    def test_multiple_predictions_independent(self, tracker, cache):
        ep1 = _make_ensemble(symbol="AAPL", n_models=2)
        ep2 = _make_ensemble(symbol="MSFT", n_models=2)
        id1 = tracker.store_prediction(ep1)
        id2 = tracker.store_prediction(ep2)

        assert id1 != id2

        session = cache._session()
        try:
            total = session.query(PredictionRow).count()
            assert total == 6  # 2*(2+1)
        finally:
            session.close()

    def test_stores_zero_models(self, tracker, cache):
        ep = _make_ensemble(n_models=0)
        eid = tracker.store_prediction(ep)

        session = cache._session()
        try:
            total = session.query(PredictionRow).count()
            assert total == 1  # just the ensemble row
            row = session.query(PredictionRow).filter(PredictionRow.id == eid).first()
            assert row.is_ensemble == 1
        finally:
            session.close()


# ── get_pending tests ───────────────────────────────────────────────────


class TestGetPending:
    def test_returns_past_due_unresolved(self, tracker):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ep = _make_ensemble(target_date=yesterday)
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        assert len(pending) > 0
        assert all(p["resolved"] == 0 for p in pending)

    def test_excludes_future_predictions(self, tracker):
        future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        ep = _make_ensemble(target_date=future)
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        assert len(pending) == 0

    def test_excludes_already_resolved(self, tracker, cache):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ep = _make_ensemble(target_date=yesterday)
        tracker.store_prediction(ep)

        # Manually mark all as resolved
        with _lock:
            session = cache._session()
            try:
                session.query(PredictionRow).update({PredictionRow.resolved: 1})
                session.commit()
            finally:
                session.close()

        pending = tracker.get_pending()
        assert len(pending) == 0

    def test_returns_dict_with_all_keys(self, tracker):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        ep = _make_ensemble(target_date=yesterday)
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        assert len(pending) > 0
        expected_keys = {
            "id", "timestamp", "symbol", "asset_class", "model_name",
            "prediction_direction", "predicted_magnitude", "confidence",
            "horizon_days", "horizon_hours", "target_date", "interval",
            "consensus_signal", "is_ensemble", "resolved",
        }
        assert expected_keys == set(pending[0].keys())

    def test_empty_db_returns_empty(self, tracker):
        assert tracker.get_pending() == []


# ── get_predictions tests ───────────────────────────────────────────────


class TestGetPredictions:
    def test_returns_all_predictions(self, tracker):
        ep = _make_ensemble(n_models=2)
        tracker.store_prediction(ep)

        results = tracker.get_predictions()
        assert len(results) == 3  # 2 models + 1 ensemble

    def test_filter_by_symbol(self, tracker):
        tracker.store_prediction(_make_ensemble(symbol="AAPL", n_models=1))
        tracker.store_prediction(_make_ensemble(symbol="MSFT", n_models=1))

        aapl = tracker.get_predictions(symbol="AAPL")
        assert all(r["symbol"] == "AAPL" for r in aapl)

    def test_filter_by_model(self, tracker):
        tracker.store_prediction(_make_ensemble(n_models=3))

        ensemble_only = tracker.get_predictions(model="ensemble")
        assert len(ensemble_only) == 1
        assert ensemble_only[0]["model_name"] == "ensemble"

    def test_filter_by_asset_class(self, tracker):
        tracker.store_prediction(_make_ensemble(asset_class=AssetClass.EQUITY, n_models=1))
        tracker.store_prediction(_make_ensemble(asset_class=AssetClass.CRYPTO, n_models=1))

        equity = tracker.get_predictions(asset_class="equity")
        assert all(r["asset_class"] == "equity" for r in equity)

    def test_limit(self, tracker):
        for i in range(5):
            tracker.store_prediction(_make_ensemble(symbol=f"SYM{i}", n_models=0))

        results = tracker.get_predictions(limit=3)
        assert len(results) == 3

    def test_ordered_by_timestamp_desc(self, tracker):
        tracker.store_prediction(_make_ensemble(n_models=0))
        tracker.store_prediction(_make_ensemble(n_models=0))

        results = tracker.get_predictions()
        for i in range(len(results) - 1):
            assert results[i]["timestamp"] >= results[i + 1]["timestamp"]

    def test_empty_db_returns_empty(self, tracker):
        assert tracker.get_predictions() == []

    def test_combined_filters(self, tracker):
        tracker.store_prediction(
            _make_ensemble(symbol="AAPL", asset_class=AssetClass.EQUITY, n_models=1)
        )
        tracker.store_prediction(
            _make_ensemble(symbol="BTC-USD", asset_class=AssetClass.CRYPTO, n_models=1)
        )

        results = tracker.get_predictions(symbol="AAPL", asset_class="equity")
        assert len(results) > 0
        assert all(r["symbol"] == "AAPL" and r["asset_class"] == "equity" for r in results)


# ── get_performance tests ───────────────────────────────────────────────


class TestGetPerformance:
    def _seed_performance(self, cache, model_name, asset_class, wins, losses, avg_conf):
        with _lock:
            session = cache._session()
            try:
                session.add(ModelPerformanceRow(
                    model_name=model_name,
                    asset_class=asset_class,
                    total_predictions=wins + losses,
                    wins=wins,
                    losses=losses,
                    win_rate=wins / (wins + losses) if (wins + losses) else 0,
                    avg_confidence=avg_conf,
                    avg_error=0.02,
                    last_updated=datetime.now(),
                ))
                session.commit()
            finally:
                session.close()

    def test_returns_model_performance_objects(self, tracker, cache):
        self._seed_performance(cache, "gradient_boost", "equity", 7, 3, 0.75)

        perfs = tracker.get_performance()
        assert len(perfs) == 1
        assert isinstance(perfs[0], ModelPerformance)

    def test_filter_by_model_name(self, tracker, cache):
        self._seed_performance(cache, "gradient_boost", "equity", 7, 3, 0.75)
        self._seed_performance(cache, "prophet", "equity", 5, 5, 0.6)

        perfs = tracker.get_performance(model_name="gradient_boost")
        assert len(perfs) == 1
        assert perfs[0].model_name == "gradient_boost"

    def test_filter_by_asset_class(self, tracker, cache):
        self._seed_performance(cache, "gradient_boost", "equity", 7, 3, 0.75)
        self._seed_performance(cache, "gradient_boost", "crypto", 4, 6, 0.55)

        perfs = tracker.get_performance(asset_class="equity")
        assert len(perfs) == 1
        assert perfs[0].asset_class == AssetClass.EQUITY

    def test_correct_win_rate(self, tracker, cache):
        self._seed_performance(cache, "model_a", "equity", 8, 2, 0.8)

        perf = tracker.get_performance(model_name="model_a")[0]
        assert abs(perf.win_rate - 0.8) < 1e-6

    def test_empty_returns_empty(self, tracker):
        assert tracker.get_performance() == []


# ── update_performance tests ────────────────────────────────────────────


class TestUpdatePerformance:
    def _seed_resolved(self, cache, model_name, asset_class, correct_flags):
        """Insert resolved predictions + outcomes for performance calc."""
        with _lock:
            session = cache._session()
            try:
                for correct in correct_flags:
                    pred = PredictionRow(
                        timestamp=datetime.now(),
                        symbol="TEST",
                        asset_class=asset_class,
                        model_name=model_name,
                        prediction_direction="up",
                        predicted_magnitude=0.05,
                        confidence=0.7,
                        horizon_days=5,
                        target_date="2025-01-01",
                        is_ensemble=0,
                        resolved=1,
                    )
                    session.add(pred)
                    session.flush()

                    session.add(PredictionOutcomeRow(
                        prediction_id=pred.id,
                        actual_direction="up" if correct else "down",
                        actual_magnitude=0.04,
                        correct=1 if correct else 0,
                        error_pct=0.01,
                        resolved_at=datetime.now(),
                    ))
                session.commit()
            finally:
                session.close()

    def test_creates_performance_row(self, tracker, cache):
        self._seed_resolved(cache, "gradient_boost", "equity", [True, True, False])
        tracker.update_performance("gradient_boost", "equity")

        perfs = tracker.get_performance(model_name="gradient_boost")
        assert len(perfs) == 1

    def test_correct_win_rate(self, tracker, cache):
        self._seed_resolved(cache, "model_x", "equity", [True, True, True, False])
        tracker.update_performance("model_x", "equity")

        perf = tracker.get_performance(model_name="model_x")[0]
        assert abs(perf.win_rate - 0.75) < 1e-6

    def test_correct_wins_losses(self, tracker, cache):
        self._seed_resolved(cache, "model_y", "crypto", [True, False, False])
        tracker.update_performance("model_y", "crypto")

        perf = tracker.get_performance(model_name="model_y")[0]
        assert perf.wins == 1
        assert perf.losses == 2
        assert perf.total_predictions == 3

    def test_upsert_updates_existing(self, tracker, cache):
        self._seed_resolved(cache, "model_z", "equity", [True])
        tracker.update_performance("model_z", "equity")

        perf1 = tracker.get_performance(model_name="model_z")[0]
        assert perf1.total_predictions == 1

        # Add more resolved predictions
        self._seed_resolved(cache, "model_z", "equity", [True, False])
        tracker.update_performance("model_z", "equity")

        perf2 = tracker.get_performance(model_name="model_z")[0]
        assert perf2.total_predictions == 3

    def test_no_resolved_does_nothing(self, tracker, cache):
        tracker.update_performance("nonexistent", "equity")
        assert tracker.get_performance() == []


# ── Intraday tracker tests ─────────────────────────────────────────────


class TestIntradayTracker:
    def test_store_prediction_with_interval(self, tracker, cache):
        ep = _make_ensemble(target_date="2020-01-01 14:00:00")
        # Override interval on the EnsemblePrediction
        ep.interval = "1h"
        eid = tracker.store_prediction(ep)

        session = cache._session()
        try:
            row = session.query(PredictionRow).filter(PredictionRow.id == eid).first()
            assert row is not None
            assert row.interval == "1h"
        finally:
            session.close()

        # Also verify via get_predictions dict
        preds = tracker.get_predictions(model="ensemble")
        ensemble_pred = [p for p in preds if p["id"] == eid][0]
        assert ensemble_pred["interval"] == "1h"

    def test_get_pending_intraday(self, tracker):
        # Store a prediction with a past intraday target_date
        ep = _make_ensemble(target_date="2020-01-01 14:00:00")
        ep.interval = "1h"
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        # 2020-01-01 14:00:00 is in the past, so it should be pending
        assert len(pending) > 0
        # Verify the intraday target_date is preserved
        intraday = [p for p in pending if "14:00:00" in p["target_date"]]
        assert len(intraday) > 0
        assert intraday[0]["interval"] == "1h"
