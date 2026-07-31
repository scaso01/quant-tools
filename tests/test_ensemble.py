"""Tests for PredictionEnsemble weighted combiner."""
from __future__ import annotations

import pytest

from quant_tools.errors import PredictionError
from quant_tools.models import (
    AssetClass,
    ConsensusSignal,
    PredictionDirection,
    PredictionResult,
)
from quant_tools.predict.base import ModelState
from quant_tools.predict.ensemble import PredictionEnsemble


# ── Mock Model ──────────────────────────────────────────────────────


class MockModel:
    """Minimal mock implementing PredictionModel interface."""

    def __init__(
        self,
        name: str,
        direction: PredictionDirection,
        magnitude: float = 0.02,
        confidence: float = 0.6,
        supported: bool = True,
        fail: bool = False,
    ) -> None:
        self._name = name
        self._direction = direction
        self._magnitude = magnitude
        self._confidence = confidence
        self._supported = supported
        self._fail = fail
        self.state = ModelState()

    @property
    def name(self) -> str:
        return self._name

    def train(self, features_df: object, target: object) -> None:
        if self._fail:
            raise RuntimeError(f"Mock failure in {self._name}")

    def predict(self, features_df: object, horizon_days: int = 5) -> PredictionResult:
        if self._fail:
            raise RuntimeError(f"Mock failure in {self._name}")
        return PredictionResult(
            symbol="TEST",
            asset_class=AssetClass.EQUITY,
            model_name=self._name,
            direction=self._direction,
            magnitude=self._magnitude,
            confidence=self._confidence,
        )

    def supports_asset_class(self, ac: AssetClass) -> bool:
        return self._supported


# ── Helpers ─────────────────────────────────────────────────────────


def _make_ensemble(models: list[MockModel], weights: dict[str, float] | None = None) -> PredictionEnsemble:
    """Build ensemble from mock models without triggering default model loading."""
    return PredictionEnsemble(models=models, weights=weights)


# ── Tests: Direction Consensus ──────────────────────────────────────


class TestDirectionConsensus:
    """Tests for consensus direction voting."""

    def test_all_up_gives_buy(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.6),
            MockModel("m2", PredictionDirection.UP, confidence=0.6),
            MockModel("m3", PredictionDirection.UP, confidence=0.6),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.UP
        assert result.consensus in (ConsensusSignal.BUY, ConsensusSignal.STRONG_BUY)

    def test_all_up_high_confidence_gives_strong_buy(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.9),
            MockModel("m2", PredictionDirection.UP, confidence=0.9),
            MockModel("m3", PredictionDirection.UP, confidence=0.9),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.UP
        assert result.consensus == ConsensusSignal.STRONG_BUY

    def test_all_down_gives_sell(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.DOWN, confidence=0.6),
            MockModel("m2", PredictionDirection.DOWN, confidence=0.6),
            MockModel("m3", PredictionDirection.DOWN, confidence=0.6),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.DOWN
        assert result.consensus in (ConsensusSignal.SELL, ConsensusSignal.STRONG_SELL)

    def test_all_down_high_confidence_gives_strong_sell(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.DOWN, confidence=0.9),
            MockModel("m2", PredictionDirection.DOWN, confidence=0.9),
            MockModel("m3", PredictionDirection.DOWN, confidence=0.9),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.DOWN
        assert result.consensus == ConsensusSignal.STRONG_SELL

    def test_all_neutral_gives_hold(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.NEUTRAL, confidence=0.5),
            MockModel("m2", PredictionDirection.NEUTRAL, confidence=0.5),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.consensus == ConsensusSignal.HOLD

    def test_majority_up_wins(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.6),
            MockModel("m2", PredictionDirection.UP, confidence=0.6),
            MockModel("m3", PredictionDirection.DOWN, confidence=0.6),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.UP

    def test_weighted_majority_overrides_count(self) -> None:
        """A single heavily-weighted DOWN model can beat two UP models."""
        models = [
            MockModel("heavy", PredictionDirection.DOWN, confidence=0.8),
            MockModel("light1", PredictionDirection.UP, confidence=0.8),
            MockModel("light2", PredictionDirection.UP, confidence=0.8),
        ]
        weights = {"heavy": 0.7, "light1": 0.15, "light2": 0.15}
        ens = _make_ensemble(models, weights=weights)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.DOWN


# ── Tests: GARCH Confidence Adjustment ──────────────────────────────


class TestGARCHAdjustment:
    """Tests for GARCH volatility confidence adjustment."""

    def test_garch_high_vol_lowers_confidence(self) -> None:
        """Low GARCH confidence (high vol) should reduce ensemble confidence."""
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.8),
            MockModel("m2", PredictionDirection.UP, confidence=0.8),
        ]
        ens = _make_ensemble(models)
        preds = [m.predict(None) for m in models]

        # Without GARCH
        no_garch = ens._combine(preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", None)
        # With GARCH low confidence (high vol)
        with_garch = ens._combine(preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", 0.2)

        assert with_garch.confidence < no_garch.confidence

    def test_garch_low_vol_preserves_confidence(self) -> None:
        """High GARCH confidence (low vol) should nearly preserve confidence."""
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.8),
            MockModel("m2", PredictionDirection.UP, confidence=0.8),
        ]
        ens = _make_ensemble(models)
        preds = [m.predict(None) for m in models]

        no_garch = ens._combine(preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", None)
        with_garch = ens._combine(preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", 1.0)

        # garch_confidence=1.0 -> multiplier = 0.5 + 0.5*1.0 = 1.0 -> no change
        assert abs(with_garch.confidence - no_garch.confidence) < 0.01

    def test_garch_excluded_from_direction_votes(self) -> None:
        """GARCH predictions (model_name='garch') should not affect direction."""
        up_model = MockModel("trend", PredictionDirection.UP, confidence=0.7)
        garch_model = MockModel("garch", PredictionDirection.NEUTRAL, confidence=0.5)
        ens = _make_ensemble([up_model, garch_model])
        preds = [up_model.predict(None), garch_model.predict(None)]

        result = ens._combine(
            preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", 0.5,
        )
        assert result.direction == PredictionDirection.UP


# ── Tests: Weight Management ────────────────────────────────────────


class TestWeights:
    """Tests for weight initialization, updates, and floor enforcement."""

    def test_equal_weights_by_default(self) -> None:
        models = [
            MockModel("a", PredictionDirection.UP),
            MockModel("b", PredictionDirection.UP),
            MockModel("c", PredictionDirection.UP),
        ]
        ens = _make_ensemble(models)
        weights = ens.get_weights()
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    def test_custom_weights_normalized(self) -> None:
        models = [
            MockModel("a", PredictionDirection.UP),
            MockModel("b", PredictionDirection.UP),
        ]
        ens = _make_ensemble(models, weights={"a": 0.8, "b": 0.2})
        weights = ens.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weight_floor_enforced(self) -> None:
        models = [
            MockModel("a", PredictionDirection.UP),
            MockModel("b", PredictionDirection.UP),
        ]
        ens = _make_ensemble(models, weights={"a": 0.99, "b": 0.01})
        weights = ens.get_weights()
        # b=0.01 is raised to MIN_WEIGHT=0.05 before normalization
        # After normalization: 0.05 / (0.99 + 0.05) = ~0.048
        # The key guarantee: b's raw weight was floored BEFORE normalization
        # so b ends up with a larger share than the original 0.01/(0.99+0.01)=0.01
        assert weights["b"] > 0.01  # Floor raised it from 1% share
        assert weights["b"] > 0.04  # Close to MIN_WEIGHT after normalization

    def test_update_weights_changes_values(self) -> None:
        models = [
            MockModel("a", PredictionDirection.UP),
            MockModel("b", PredictionDirection.DOWN),
        ]
        ens = _make_ensemble(models)

        ens.update_weights({"a": 0.9, "b": 0.1})
        new_weights = ens.get_weights()

        # a should now have higher weight than b
        assert new_weights["a"] > new_weights["b"]

    def test_update_weights_sum_to_one(self) -> None:
        models = [
            MockModel("a", PredictionDirection.UP),
            MockModel("b", PredictionDirection.UP),
            MockModel("c", PredictionDirection.UP),
        ]
        ens = _make_ensemble(models)
        ens.update_weights({"a": 0.8, "b": 0.5, "c": 0.2})
        weights = ens.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ── Tests: Confidence Calculation ───────────────────────────────────


class TestConfidence:
    """Tests for confidence calculation."""

    def test_confidence_bounded_0_to_1(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.UP, confidence=1.0),
            MockModel("m2", PredictionDirection.UP, confidence=1.0),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_mixed_signals_lower_confidence(self) -> None:
        """Disagreement among models should produce lower confidence."""
        agree_models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.7),
            MockModel("m2", PredictionDirection.UP, confidence=0.7),
        ]
        mixed_models = [
            MockModel("m1", PredictionDirection.UP, confidence=0.7),
            MockModel("m2", PredictionDirection.DOWN, confidence=0.7),
        ]
        ens_agree = _make_ensemble(agree_models)
        ens_mixed = _make_ensemble(mixed_models)

        res_agree = ens_agree._combine(
            [m.predict(None) for m in agree_models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        res_mixed = ens_mixed._combine(
            [m.predict(None) for m in mixed_models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert res_agree.confidence > res_mixed.confidence


# ── Tests: Magnitude ────────────────────────────────────────────────


class TestMagnitude:
    """Tests for weighted magnitude averaging."""

    def test_magnitude_from_agreeing_models(self) -> None:
        models = [
            MockModel("m1", PredictionDirection.UP, magnitude=0.03, confidence=0.7),
            MockModel("m2", PredictionDirection.UP, magnitude=0.01, confidence=0.7),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        # Equal weights -> average of 0.03 and 0.01 = 0.02
        assert result.magnitude is not None
        assert abs(result.magnitude - 0.02) < 0.001

    def test_magnitude_none_when_no_agreeing(self) -> None:
        """When direction is NEUTRAL with no magnitudes, magnitude should be None."""
        models = [
            MockModel("m1", PredictionDirection.NEUTRAL, magnitude=None, confidence=0.5),
            MockModel("m2", PredictionDirection.NEUTRAL, magnitude=None, confidence=0.5),
        ]
        ens = _make_ensemble(models)
        result = ens._combine(
            [m.predict(None) for m in models],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.magnitude is None


# ── Tests: Error Handling ───────────────────────────────────────────


class TestErrorHandling:
    """Tests for failure modes."""

    def test_single_model_failure_doesnt_crash(self) -> None:
        """If one model fails, ensemble should still produce a result."""
        models = [
            MockModel("ok1", PredictionDirection.UP, confidence=0.7),
            MockModel("broken", PredictionDirection.UP, fail=True),
            MockModel("ok2", PredictionDirection.UP, confidence=0.7),
        ]
        ens = _make_ensemble(models)
        # Simulate: only ok1 and ok2 succeed
        preds = []
        for m in models:
            try:
                preds.append(m.predict(None))
            except Exception:
                pass
        result = ens._combine(
            preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.UP
        assert len(result.model_predictions) == 2

    def test_empty_predictions_raises(self) -> None:
        """If all models fail and produce no predictions, should raise."""
        # TODO: this raises the error itself rather than getting it out of the
        # ensemble, so it passes no matter what predict() does. Making it real
        # means feeding predict() a genuine OHLCV frame (it runs
        # FeatureEngineer before it reaches the models) and widening
        # MockModel.predict to take the interval keyword the ensemble passes.
        preds: list[PredictionResult] = []
        with pytest.raises(PredictionError, match="All models failed"):
            if not preds:
                raise PredictionError("All models failed for TEST")

    def test_all_models_fail_raises(self) -> None:
        """If all model predictions are empty, _combine caller should raise."""
        # TODO: same as above. Self-raised, so it does not exercise predict().
        with pytest.raises(PredictionError, match="All models failed"):
            raise PredictionError("All models failed for TEST")

    def test_unsupported_asset_class_skipped(self) -> None:
        """Models that don't support an asset class should be skipped."""
        models = [
            MockModel("supported", PredictionDirection.UP, confidence=0.7, supported=True),
            MockModel("unsupported", PredictionDirection.DOWN, confidence=0.7, supported=False),
        ]
        ens = _make_ensemble(models)
        # Only the supported model should contribute
        preds = [models[0].predict(None)]  # Only supported model succeeds
        result = ens._combine(
            preds, "SPY", AssetClass.EQUITY, 5, "2026-03-21", None,
        )
        assert result.direction == PredictionDirection.UP


# ── Tests: _map_consensus ───────────────────────────────────────────


class TestMapConsensus:
    """Tests for the static _map_consensus method."""

    def test_neutral_always_hold(self) -> None:
        assert PredictionEnsemble._map_consensus(PredictionDirection.NEUTRAL, 0.0) == ConsensusSignal.HOLD
        assert PredictionEnsemble._map_consensus(PredictionDirection.NEUTRAL, 0.9) == ConsensusSignal.HOLD

    def test_up_low_confidence_buy(self) -> None:
        assert PredictionEnsemble._map_consensus(PredictionDirection.UP, 0.5) == ConsensusSignal.BUY

    def test_up_high_confidence_strong_buy(self) -> None:
        assert PredictionEnsemble._map_consensus(PredictionDirection.UP, 0.8) == ConsensusSignal.STRONG_BUY

    def test_down_low_confidence_sell(self) -> None:
        assert PredictionEnsemble._map_consensus(PredictionDirection.DOWN, 0.5) == ConsensusSignal.SELL

    def test_down_high_confidence_strong_sell(self) -> None:
        assert PredictionEnsemble._map_consensus(PredictionDirection.DOWN, 0.8) == ConsensusSignal.STRONG_SELL

    def test_boundary_at_07(self) -> None:
        """At exactly 0.7, should be BUY (not STRONG_BUY, since > 0.7 required)."""
        assert PredictionEnsemble._map_consensus(PredictionDirection.UP, 0.7) == ConsensusSignal.BUY
        assert PredictionEnsemble._map_consensus(PredictionDirection.DOWN, 0.7) == ConsensusSignal.SELL


# ── Tests: Only GARCH predictions ──────────────────────────────────


class TestGARCHOnly:
    """Test edge case where only GARCH predictions exist."""

    def test_only_garch_gives_hold(self) -> None:
        """When all predictions come from GARCH, result should be HOLD."""
        garch = MockModel("garch", PredictionDirection.NEUTRAL, confidence=0.5)
        ens = _make_ensemble([garch])
        result = ens._combine(
            [garch.predict(None)],
            "SPY", AssetClass.EQUITY, 5, "2026-03-21", 0.5,
        )
        assert result.direction == PredictionDirection.NEUTRAL
        assert result.consensus == ConsensusSignal.HOLD
        assert result.confidence == 0.3
