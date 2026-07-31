"""Tests for factors/rebalance.py."""

from __future__ import annotations

import pytest

from quant_tools.factors.rebalance import (
    REGIME_TILTS,
    _apply_regime_tilt,
    get_recommendations,
)
from quant_tools.models import FactorScore, MacroRegime, RebalanceRecommendation


def _make_factor_scores(n: int = 15) -> list[FactorScore]:
    """Create sample FactorScore objects for testing.

    Args:
        n: Number of scores to generate.

    Returns:
        List of FactorScore with varying z-scores.
    """
    scores = []
    for i in range(n):
        scores.append(FactorScore(
            symbol=f"SYM{i:02d}",
            name=f"Stock {i}",
            value_z=1.0 - i * 0.15,
            momentum_z=-0.5 + i * 0.1,
            quality_z=0.8 - i * 0.12,
            composite_z=0.0,
            rank=0,
            sector="Technology" if i % 2 == 0 else "Healthcare",
        ))
    return scores


# =============================================================================
# Tilt Weight Validation
# =============================================================================


class TestTiltWeights:
    """Verify tilt weights sum to 1.0 for all regimes."""

    @pytest.mark.parametrize("regime", list(MacroRegime))
    def test_tilts_sum_to_one(self, regime: MacroRegime):
        """All tilt weights for a regime should sum to 1.0."""
        tilts = REGIME_TILTS[regime]
        total = sum(tilts.values())
        assert abs(total - 1.0) < 1e-10, f"{regime.value} tilts sum to {total}"

    @pytest.mark.parametrize("regime", list(MacroRegime))
    def test_all_tilts_positive(self, regime: MacroRegime):
        """All tilt weights should be positive."""
        tilts = REGIME_TILTS[regime]
        for factor, weight in tilts.items():
            assert weight > 0, f"{regime.value}/{factor} has non-positive weight {weight}"

    @pytest.mark.parametrize("regime", list(MacroRegime))
    def test_all_factors_present(self, regime: MacroRegime):
        """Each regime should have value, momentum, quality, defensive tilts."""
        tilts = REGIME_TILTS[regime]
        for factor in ("value", "momentum", "quality", "defensive"):
            assert factor in tilts, f"{regime.value} missing {factor} tilt"


# =============================================================================
# _apply_regime_tilt
# =============================================================================


class TestApplyRegimeTilt:
    """Test regime tilt application to factor scores."""

    def test_reranks_by_tilted_composite(self):
        """Applying tilts should change composite scores and re-rank."""
        scores = _make_factor_scores(5)
        tilts = REGIME_TILTS[MacroRegime.EXPANSION]
        adjusted = _apply_regime_tilt(scores, tilts)

        assert len(adjusted) == 5
        # Should be sorted descending by composite_z
        for i in range(len(adjusted) - 1):
            assert adjusted[i].composite_z >= adjusted[i + 1].composite_z

    def test_ranks_are_sequential(self):
        """Ranks should be 1 through N."""
        scores = _make_factor_scores(10)
        tilts = REGIME_TILTS[MacroRegime.CONTRACTION]
        adjusted = _apply_regime_tilt(scores, tilts)

        ranks = [fs.rank for fs in adjusted]
        assert sorted(ranks) == list(range(1, 11))

    def test_original_zscores_preserved(self):
        """Individual factor z-scores should be preserved (only composite changes)."""
        # _apply_regime_tilt builds a new list, so comparing against `scores`
        # afterwards is a real check and needs no snapshot.
        scores = _make_factor_scores(3)
        tilts = REGIME_TILTS[MacroRegime.RECOVERY]
        adjusted = _apply_regime_tilt(scores, tilts)

        # value_z should be unchanged for each symbol
        for orig_fs in scores:
            matching = [a for a in adjusted if a.symbol == orig_fs.symbol]
            assert len(matching) == 1
            assert matching[0].value_z == orig_fs.value_z
            assert matching[0].momentum_z == orig_fs.momentum_z
            assert matching[0].quality_z == orig_fs.quality_z

    def test_contraction_favors_defensive(self):
        """In contraction, high quality + low momentum should rank higher."""
        # Create two stocks: one defensive (high quality, low momentum), one aggressive
        defensive = FactorScore(
            symbol="DEF", name="Defensive Co",
            value_z=0.5, momentum_z=-1.0, quality_z=1.5,
            composite_z=0.0, sector="Utilities",
        )
        aggressive = FactorScore(
            symbol="AGG", name="Aggressive Co",
            value_z=-0.5, momentum_z=1.5, quality_z=-0.5,
            composite_z=0.0, sector="Technology",
        )

        tilts = REGIME_TILTS[MacroRegime.CONTRACTION]
        adjusted = _apply_regime_tilt([defensive, aggressive], tilts)

        # Defensive stock should rank #1 in contraction
        assert adjusted[0].symbol == "DEF"


# =============================================================================
# get_recommendations
# =============================================================================


class TestGetRecommendations:
    """Test full recommendation generation."""

    @pytest.mark.parametrize("regime", list(MacroRegime))
    def test_returns_recommendation(self, regime: MacroRegime):
        """Each regime should produce a valid RebalanceRecommendation."""
        scores = _make_factor_scores(15)
        rec = get_recommendations(regime, scores, top_n=5)

        assert isinstance(rec, RebalanceRecommendation)
        assert rec.regime == regime
        assert len(rec.top_picks) == 5
        assert rec.rationale != ""

    @pytest.mark.parametrize("regime", list(MacroRegime))
    def test_recommendation_tilts_sum_to_one(self, regime: MacroRegime):
        """Recommendation tilts should sum to 1.0."""
        scores = _make_factor_scores(10)
        rec = get_recommendations(regime, scores, top_n=5)

        total = rec.value_tilt + rec.momentum_tilt + rec.quality_tilt + rec.defensive_tilt
        assert abs(total - 1.0) < 1e-10

    def test_top_n_limits_output(self):
        """Should return exactly top_n picks."""
        scores = _make_factor_scores(20)
        rec = get_recommendations(MacroRegime.EXPANSION, scores, top_n=3)
        assert len(rec.top_picks) == 3

    def test_top_n_exceeds_scores(self):
        """If top_n > len(scores), return all available."""
        scores = _make_factor_scores(3)
        rec = get_recommendations(MacroRegime.EXPANSION, scores, top_n=10)
        assert len(rec.top_picks) == 3

    def test_expansion_tilts_match_config(self):
        """Expansion tilts should match the REGIME_TILTS config."""
        scores = _make_factor_scores(5)
        rec = get_recommendations(MacroRegime.EXPANSION, scores)

        assert rec.momentum_tilt == 0.35
        assert rec.quality_tilt == 0.30
        assert rec.value_tilt == 0.20
        assert rec.defensive_tilt == 0.15

    def test_contraction_tilts_match_config(self):
        """Contraction tilts should match the REGIME_TILTS config."""
        scores = _make_factor_scores(5)
        rec = get_recommendations(MacroRegime.CONTRACTION, scores)

        assert rec.defensive_tilt == 0.40
        assert rec.quality_tilt == 0.30
        assert rec.value_tilt == 0.20
        assert rec.momentum_tilt == 0.10

    def test_top_picks_are_ranked(self):
        """Top picks should have sequential ranks starting from 1."""
        scores = _make_factor_scores(15)
        rec = get_recommendations(MacroRegime.RECOVERY, scores, top_n=5)

        for i, pick in enumerate(rec.top_picks):
            assert pick.rank == i + 1

    def test_empty_scores_returns_empty_picks(self):
        """Empty factor scores should produce empty top_picks."""
        rec = get_recommendations(MacroRegime.EXPANSION, [], top_n=5)
        assert len(rec.top_picks) == 0
        assert rec.regime == MacroRegime.EXPANSION
