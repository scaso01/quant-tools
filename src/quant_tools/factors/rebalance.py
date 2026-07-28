"""Factor tilt recommendations based on macro regime."""

from __future__ import annotations

import logging

from quant_tools.models import FactorScore, MacroRegime, RebalanceRecommendation

log = logging.getLogger("quant_tools")

# Regime-based factor tilts
REGIME_TILTS: dict[MacroRegime, dict[str, float]] = {
    MacroRegime.EXPANSION: {
        "momentum": 0.35,
        "quality": 0.30,
        "value": 0.20,
        "defensive": 0.15,
    },
    MacroRegime.SLOWDOWN: {
        "quality": 0.35,
        "defensive": 0.30,
        "value": 0.25,
        "momentum": 0.10,
    },
    MacroRegime.CONTRACTION: {
        "defensive": 0.40,
        "quality": 0.30,
        "value": 0.20,
        "momentum": 0.10,
    },
    MacroRegime.RECOVERY: {
        "value": 0.35,
        "momentum": 0.30,
        "quality": 0.20,
        "defensive": 0.15,
    },
}

# Rationale text per regime
REGIME_RATIONALE: dict[MacroRegime, str] = {
    MacroRegime.EXPANSION: (
        "Expansion: Favor momentum and quality stocks. Growth is strong, "
        "so ride winners and lean into high-quality compounders."
    ),
    MacroRegime.SLOWDOWN: (
        "Slowdown: Shift to quality and defensive names. Growth is decelerating, "
        "so prioritize earnings stability and balance sheet strength."
    ),
    MacroRegime.CONTRACTION: (
        "Contraction: Maximum defensive posture. Protect capital with low-beta "
        "and high-quality names. Avoid momentum chasers."
    ),
    MacroRegime.RECOVERY: (
        "Recovery: Load up on value and momentum. Beaten-down stocks with strong "
        "fundamentals tend to lead early-cycle rallies."
    ),
}


def _apply_regime_tilt(
    factor_scores: list[FactorScore],
    tilts: dict[str, float],
) -> list[FactorScore]:
    """Re-score factor scores using regime-specific tilts.

    Args:
        factor_scores: Original factor scores.
        tilts: Dict mapping factor names to tilt weights.

    Returns:
        New list of FactorScore with adjusted composite_z, sorted descending.
    """
    adjusted: list[FactorScore] = []
    for fs in factor_scores:
        # Defensive tilt is inverse of momentum + quality (low-vol proxy)
        defensive_z = (fs.quality_z - fs.momentum_z) / 2

        new_composite = (
            tilts.get("value", 0.0) * fs.value_z
            + tilts.get("momentum", 0.0) * fs.momentum_z
            + tilts.get("quality", 0.0) * fs.quality_z
            + tilts.get("defensive", 0.0) * defensive_z
        )

        adjusted.append(FactorScore(
            symbol=fs.symbol,
            name=fs.name,
            value_z=fs.value_z,
            momentum_z=fs.momentum_z,
            quality_z=fs.quality_z,
            composite_z=new_composite,
            sector=fs.sector,
        ))

    # Sort by adjusted composite, assign ranks
    adjusted.sort(key=lambda f: f.composite_z, reverse=True)
    for rank, fs in enumerate(adjusted, start=1):
        fs.rank = rank

    return adjusted


def get_recommendations(
    regime: MacroRegime,
    factor_scores: list[FactorScore],
    top_n: int = 10,
) -> RebalanceRecommendation:
    """Generate factor tilt recommendations for current regime.

    Args:
        regime: Current macro regime.
        factor_scores: Pre-computed factor scores for the universe.
        top_n: Number of top picks to return.

    Returns:
        RebalanceRecommendation with tilts and top picks.
    """
    tilts = REGIME_TILTS[regime]

    log.info(
        "Generating recommendations for %s regime: value=%.0f%%, momentum=%.0f%%, "
        "quality=%.0f%%, defensive=%.0f%%",
        regime.value,
        tilts["value"] * 100,
        tilts["momentum"] * 100,
        tilts["quality"] * 100,
        tilts["defensive"] * 100,
    )

    adjusted = _apply_regime_tilt(factor_scores, tilts)
    top_picks = adjusted[:top_n]

    return RebalanceRecommendation(
        regime=regime,
        value_tilt=tilts["value"],
        momentum_tilt=tilts["momentum"],
        quality_tilt=tilts["quality"],
        defensive_tilt=tilts["defensive"],
        top_picks=top_picks,
        rationale=REGIME_RATIONALE[regime],
    )
