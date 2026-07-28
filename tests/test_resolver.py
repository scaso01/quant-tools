"""Tests for PredictionResolver — resolution against actual market data."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.cache import CacheManager, PredictionOutcomeRow, PredictionRow, _lock
from quant_tools.models import (
    AssetClass,
    ConsensusSignal,
    EnsemblePrediction,
    PredictionDirection,
    PredictionOutcome,
    PredictionResult,
)
from quant_tools.predict.resolver import PredictionResolver
from quant_tools.predict.tracker import PredictionTracker


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_db):
    return CacheManager(db_path=tmp_db)


@pytest.fixture
def tracker(cache):
    return PredictionTracker(cache_manager=cache)


@pytest.fixture
def mock_registry():
    """Mock ProviderRegistry that returns synthetic price data."""
    registry = MagicMock()
    return registry


@pytest.fixture
def resolver(tracker, mock_registry):
    return PredictionResolver(tracker=tracker, registry=mock_registry)


def _make_price_df(start_price: float, end_price: float, n_days: int = 20) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame with linear interpolation."""
    dates = pd.bdate_range(start=datetime.now() - timedelta(days=n_days + 5), periods=n_days)
    close = np.linspace(start_price, end_price, n_days)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.full(n_days, 1_000_000),
    }, index=dates)


def _make_ensemble(
    symbol: str = "AAPL",
    direction: PredictionDirection = PredictionDirection.UP,
    magnitude: float = 0.05,
    target_date: str | None = None,
) -> EnsemblePrediction:
    """Helper ensemble with 1 model sub-prediction."""
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    now = datetime.now()
    return EnsemblePrediction(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        consensus=ConsensusSignal.BUY,
        confidence=0.8,
        direction=direction,
        magnitude=magnitude,
        horizon_days=5,
        target_date=target_date,
        model_predictions=[
            PredictionResult(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                model_name="gradient_boost",
                direction=direction,
                magnitude=magnitude,
                confidence=0.8,
                horizon_days=5,
                target_date=target_date,
                timestamp=now,
            ),
        ],
        timestamp=now,
    )


def _store_and_get_pending(tracker, **kwargs) -> list[dict]:
    """Store prediction and return pending list."""
    ep = _make_ensemble(**kwargs)
    tracker.store_prediction(ep)
    return tracker.get_pending()


# ── resolve_prediction tests ────────────────────────────────────────────


class TestResolvePrediction:
    def test_resolves_correct_up(self, resolver, tracker, mock_registry):
        """Prediction=UP, actual goes UP -> correct=True."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.UP)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is not None
        assert outcome.correct is True
        assert outcome.actual_direction == PredictionDirection.UP

    def test_resolves_incorrect_up(self, resolver, tracker, mock_registry):
        """Prediction=UP, actual goes DOWN -> correct=False."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 90)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.UP)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is not None
        assert outcome.correct is False

    def test_resolves_correct_down(self, resolver, tracker, mock_registry):
        """Prediction=DOWN, actual goes DOWN -> correct=True."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 90)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.DOWN)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is not None
        assert outcome.correct is True
        assert outcome.actual_direction == PredictionDirection.DOWN

    def test_resolves_neutral(self, resolver, tracker, mock_registry):
        """When price barely moves, direction=NEUTRAL."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 100.05)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.NEUTRAL)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is not None
        assert outcome.actual_direction == PredictionDirection.NEUTRAL
        assert outcome.correct is True

    def test_stores_outcome_row(self, resolver, tracker, mock_registry, cache):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.UP)
        resolver.resolve_prediction(pending[0])

        with _lock:
            session = cache._session()
            try:
                outcomes = session.query(PredictionOutcomeRow).all()
                assert len(outcomes) == 1
                assert outcomes[0].prediction_id == pending[0]["id"]
            finally:
                session.close()

    def test_marks_prediction_resolved(self, resolver, tracker, mock_registry, cache):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.UP)
        resolver.resolve_prediction(pending[0])

        with _lock:
            session = cache._session()
            try:
                pred = session.query(PredictionRow).filter(
                    PredictionRow.id == pending[0]["id"]
                ).first()
                assert pred.resolved == 1
            finally:
                session.close()

    def test_returns_prediction_outcome_type(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        pending = _store_and_get_pending(tracker, direction=PredictionDirection.UP)

        outcome = resolver.resolve_prediction(pending[0])
        assert isinstance(outcome, PredictionOutcome)

    def test_calculates_error_pct(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        pending = _store_and_get_pending(
            tracker, direction=PredictionDirection.UP, magnitude=0.05
        )

        outcome = resolver.resolve_prediction(pending[0])
        # actual_magnitude ~0.1 (10% gain), predicted 0.05 -> error ~0.05
        assert outcome.error_pct > 0

    def test_handles_no_price_data(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = pd.DataFrame()
        pending = _store_and_get_pending(tracker)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is None

    def test_handles_none_price_data(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = None
        pending = _store_and_get_pending(tracker)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is None

    def test_handles_fetch_exception(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.side_effect = RuntimeError("API error")
        pending = _store_and_get_pending(tracker)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is None

    def test_handles_single_price_point(self, resolver, tracker, mock_registry):
        """Only 1 data point -> insufficient data -> returns None."""
        df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99],
            "Close": [100], "Volume": [1000],
        }, index=[datetime.now()])
        mock_registry.fetch_ohlcv.return_value = df
        pending = _store_and_get_pending(tracker)

        outcome = resolver.resolve_prediction(pending[0])
        assert outcome is None


# ── resolve_all_pending tests ───────────────────────────────────────────


class TestResolveAllPending:
    def test_resolves_batch(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        for sym in ["AAPL", "MSFT", "GOOG"]:
            ep = _make_ensemble(symbol=sym)
            tracker.store_prediction(ep)

        outcomes = resolver.resolve_all_pending()
        # Each ensemble has 1 model + 1 ensemble row = 2 pending per symbol
        # All past-due so all should be resolved
        assert len(outcomes) > 0

    def test_empty_pending_returns_empty(self, resolver, tracker):
        outcomes = resolver.resolve_all_pending()
        assert outcomes == []

    def test_updates_performance_after_resolve(self, resolver, tracker, mock_registry, cache):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        ep = _make_ensemble(direction=PredictionDirection.UP)
        tracker.store_prediction(ep)

        resolver.resolve_all_pending()

        perfs = tracker.get_performance()
        # Should have performance rows for gradient_boost and ensemble
        assert len(perfs) > 0

    def test_partial_resolution(self, resolver, tracker, mock_registry):
        """First call succeeds, second fails -> 1 outcome."""
        call_count = [0]
        def side_effect(symbol, **kwargs):
            call_count[0] += 1
            if symbol == "FAIL":
                return pd.DataFrame()
            return _make_price_df(100, 110)

        mock_registry.fetch_ohlcv.side_effect = side_effect

        tracker.store_prediction(_make_ensemble(symbol="AAPL"))
        tracker.store_prediction(_make_ensemble(symbol="FAIL"))

        outcomes = resolver.resolve_all_pending()
        # AAPL predictions resolve, FAIL predictions don't
        successful = [o for o in outcomes if o is not None]
        assert len(successful) > 0
        # At least some predictions from FAIL should have failed
        assert len(successful) < len(tracker.get_predictions())

    def test_returns_prediction_outcome_list(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        tracker.store_prediction(_make_ensemble())

        outcomes = resolver.resolve_all_pending()
        assert all(isinstance(o, PredictionOutcome) for o in outcomes)

    def test_resolved_not_pending_again(self, resolver, tracker, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)
        tracker.store_prediction(_make_ensemble())

        resolver.resolve_all_pending()
        # Second call should find nothing pending
        outcomes2 = resolver.resolve_all_pending()
        assert outcomes2 == []


# ── _calculate_actual tests ─────────────────────────────────────────────


class TestCalculateActual:
    def test_up_direction(self, resolver, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 120)
        direction, magnitude = resolver._calculate_actual("AAPL", "equity", datetime.now(), "2025-01-10")
        assert direction == PredictionDirection.UP
        assert magnitude > 0

    def test_down_direction(self, resolver, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 80)
        direction, magnitude = resolver._calculate_actual("AAPL", "equity", datetime.now(), "2025-01-10")
        assert direction == PredictionDirection.DOWN
        assert magnitude < 0

    def test_neutral_direction(self, resolver, mock_registry):
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 100.05)
        direction, magnitude = resolver._calculate_actual("AAPL", "equity", datetime.now(), "2025-01-10")
        assert direction == PredictionDirection.NEUTRAL


# ── lazy registry loading ───────────────────────────────────────────────


class TestLazyRegistry:
    def test_lazy_loads_registry(self, tracker):
        """When no registry passed, _get_registry lazy-imports."""
        resolver = PredictionResolver(tracker=tracker, registry=None)
        mock_reg = MagicMock()
        import sys
        fake_mod = MagicMock(ProviderRegistry=MagicMock(return_value=mock_reg))
        with patch.dict(sys.modules, {"quant_tools.data.providers.registry": fake_mod}):
            # Force re-import by clearing cached registry
            resolver._registry = None
            reg = resolver._get_registry()
            assert reg is mock_reg

    def test_uses_injected_registry(self, tracker, mock_registry):
        resolver = PredictionResolver(tracker=tracker, registry=mock_registry)
        reg = resolver._get_registry()
        assert reg is mock_registry


# ── Intraday resolver tests ────────────────────────────────────────────


class TestResolveIntraday:
    def test_resolve_intraday_prediction(self, resolver, tracker, mock_registry):
        """Prediction with datetime target_date and interval='1h' resolves correctly."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)

        ep = _make_ensemble(
            direction=PredictionDirection.UP,
            target_date="2020-01-01 14:00:00",
        )
        ep.interval = "1h"
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        assert len(pending) > 0

        # Find the ensemble row
        ensemble_pred = [p for p in pending if p["model_name"] == "ensemble"][0]
        assert ensemble_pred["interval"] == "1h"

        outcome = resolver.resolve_prediction(ensemble_pred)
        assert outcome is not None
        assert outcome.correct is True

        # Verify fetch_ohlcv was called with the intraday interval
        call_kwargs = mock_registry.fetch_ohlcv.call_args[1]
        assert call_kwargs.get("interval") == "1h"

    def test_resolve_daily_backward_compat(self, resolver, tracker, mock_registry):
        """Daily prediction with date-only target_date still resolves correctly."""
        mock_registry.fetch_ohlcv.return_value = _make_price_df(100, 110)

        ep = _make_ensemble(
            direction=PredictionDirection.UP,
            target_date="2020-01-01",
        )
        # Default interval is "1d" -- no override needed
        tracker.store_prediction(ep)

        pending = tracker.get_pending()
        assert len(pending) > 0

        ensemble_pred = [p for p in pending if p["model_name"] == "ensemble"][0]
        assert ensemble_pred["interval"] == "1d"

        outcome = resolver.resolve_prediction(ensemble_pred)
        assert outcome is not None
        assert outcome.correct is True

        # Verify fetch_ohlcv was NOT called with an interval kwarg
        # (daily mode uses just period, no interval)
        call_args = mock_registry.fetch_ohlcv.call_args
        assert call_args[1].get("interval") is None
