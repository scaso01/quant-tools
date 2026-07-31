"""Tests for signal explanations and signal breakdown functionality."""

from __future__ import annotations

import numpy as np

from quant_tools.models import SignalType
from quant_tools.scanner.signal_explanations import (
    SIGNAL_EXPLANATIONS,
    SignalExplanation,
    get_signal_explanation,
)
from quant_tools.scanner.signals import (
    mean_reversion_signal,
    momentum_signal,
    quality_signal,
    value_signal,
)


# ── Explanation Coverage Tests ───────────────────────────────────────


class TestSignalExplanationsCoverage:
    """Every SignalType must have an explanation entry."""

    def test_every_signal_type_has_explanation(self) -> None:
        for st in SignalType:
            assert st in SIGNAL_EXPLANATIONS, f"Missing explanation for {st}"

    def test_each_explanation_has_name_and_summary(self) -> None:
        for st, expl in SIGNAL_EXPLANATIONS.items():
            assert expl.name, f"{st} explanation has empty name"
            assert expl.summary, f"{st} explanation has empty summary"

    def test_each_explanation_has_at_least_one_component(self) -> None:
        for st, expl in SIGNAL_EXPLANATIONS.items():
            assert len(expl.components) >= 1, f"{st} has no components"

    def test_component_weights_sum_to_approximately_one(self) -> None:
        for st, expl in SIGNAL_EXPLANATIONS.items():
            total = sum(c.weight for c in expl.components)
            assert abs(total - 1.0) < 0.02, (
                f"{st} component weights sum to {total}, expected ~1.0"
            )

    def test_score_interpretation_has_at_least_three_ranges(self) -> None:
        for st, expl in SIGNAL_EXPLANATIONS.items():
            assert len(expl.score_interpretation) >= 3, (
                f"{st} has fewer than 3 score interpretation ranges"
            )

    def test_get_signal_explanation_returns_correct_type(self) -> None:
        for st in SignalType:
            result = get_signal_explanation(st)
            assert isinstance(result, SignalExplanation)
            assert result is SIGNAL_EXPLANATIONS[st]


# ── Breakdown Tests ──────────────────────────────────────────────────


def _make_uptrend(n: int = 100) -> np.ndarray:
    """Generate synthetic uptrend price data."""
    rng = np.random.default_rng(42)
    base = np.linspace(100, 130, n)
    noise = rng.normal(0, 0.5, n)
    return base + noise


def _make_volume(n: int = 100) -> np.ndarray:
    """Generate synthetic volume data."""
    rng = np.random.default_rng(42)
    return rng.uniform(1_000_000, 5_000_000, n)


class TestMomentumBreakdown:
    """Momentum signal breakdown returns expected keys and ranges."""

    def test_breakdown_keys(self) -> None:
        close = _make_uptrend()
        volume = _make_volume()
        score, breakdown = momentum_signal(close, volume, return_breakdown=True)
        expected_keys = {"rsi", "sma_crossover", "return_1m", "volume_surge"}
        assert set(breakdown.keys()) == expected_keys

    def test_breakdown_values_in_range(self) -> None:
        close = _make_uptrend()
        score, breakdown = momentum_signal(close, return_breakdown=True)
        for key in ("rsi", "sma_crossover", "return_1m"):
            assert 0 <= breakdown[key] <= 100, f"{key}={breakdown[key]} out of [0,100]"
        assert 0 <= breakdown["volume_surge"] <= 100

    def test_without_breakdown_returns_float(self) -> None:
        close = _make_uptrend()
        result = momentum_signal(close)
        assert isinstance(result, float)


class TestValueBreakdown:
    """Value signal breakdown returns expected keys and ranges."""

    def test_breakdown_keys(self) -> None:
        info = {"pe_ratio": 15, "pb_ratio": 2.0, "forward_pe": 12}
        score, breakdown = value_signal(info, return_breakdown=True)
        expected_keys = {"pe_ratio", "pb_ratio", "forward_pe_bonus"}
        assert set(breakdown.keys()) == expected_keys

    def test_breakdown_values_in_range(self) -> None:
        info = {"pe_ratio": 15, "pb_ratio": 2.0, "forward_pe": 12}
        score, breakdown = value_signal(info, return_breakdown=True)
        for key, val in breakdown.items():
            assert 0 <= val <= 100, f"{key}={val} out of [0,100]"

    def test_without_breakdown_returns_float(self) -> None:
        info = {"pe_ratio": 15}
        result = value_signal(info)
        assert isinstance(result, float)


class TestQualityBreakdown:
    """Quality signal breakdown returns expected keys and ranges."""

    def test_breakdown_keys(self) -> None:
        info = {"roe": 0.20, "profit_margin": 0.15, "debt_to_equity": 50}
        score, breakdown = quality_signal(info, return_breakdown=True)
        expected_keys = {"roe", "profit_margin", "debt_equity"}
        assert set(breakdown.keys()) == expected_keys

    def test_breakdown_values_in_range(self) -> None:
        info = {"roe": 0.20, "profit_margin": 0.15, "debt_to_equity": 50}
        score, breakdown = quality_signal(info, return_breakdown=True)
        for key, val in breakdown.items():
            assert 0 <= val <= 100, f"{key}={val} out of [0,100]"

    def test_without_breakdown_returns_float(self) -> None:
        info = {"roe": 0.20}
        result = quality_signal(info)
        assert isinstance(result, float)


class TestMeanReversionBreakdown:
    """Mean reversion signal breakdown returns expected keys and ranges."""

    def test_breakdown_keys(self) -> None:
        close = _make_uptrend()
        score, breakdown = mean_reversion_signal(close, return_breakdown=True)
        expected_keys = {"bollinger_pct_b", "rsi_extreme"}
        assert set(breakdown.keys()) == expected_keys

    def test_breakdown_values_in_range(self) -> None:
        close = _make_uptrend()
        score, breakdown = mean_reversion_signal(close, return_breakdown=True)
        for key, val in breakdown.items():
            assert 0 <= val <= 100, f"{key}={val} out of [0,100]"

    def test_without_breakdown_returns_float(self) -> None:
        close = _make_uptrend()
        result = mean_reversion_signal(close)
        assert isinstance(result, float)
