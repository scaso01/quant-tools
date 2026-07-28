"""FRED-based macro regime detection."""

from __future__ import annotations

import logging

from quant_tools.data.fred_client import (
    fetch_cpi_yoy,
    fetch_fed_funds_rate,
    fetch_gdp_growth,
    fetch_unemployment,
    fetch_yield_spread_10y2y,
)
from quant_tools.models import MacroRegime, MacroSnapshot

log = logging.getLogger("quant_tools")


def _classify_regime(
    gdp: float | None,
    unemployment: float | None,
    yield_spread: float | None,
) -> tuple[MacroRegime, dict[str, str]]:
    """Classify macro regime from indicators.

    Regime logic:
    - EXPANSION: GDP > 2%, unemployment < 5%, yield spread > 0
    - SLOWDOWN: GDP > 0% but < 2%, or yield spread narrowing
    - CONTRACTION: GDP < 0%, or unemployment rising fast
    - RECOVERY: GDP turning positive, unemployment peaked

    Args:
        gdp: Real GDP growth rate (annualized %).
        unemployment: Unemployment rate (%).
        yield_spread: 10Y-2Y Treasury spread (%).

    Returns:
        Tuple of (MacroRegime, signals dict describing reasoning).
    """
    signals: dict[str, str] = {}

    # Score each indicator
    gdp_signal = "unknown"
    unemp_signal = "unknown"
    spread_signal = "unknown"

    if gdp is not None:
        if gdp > 2.0:
            gdp_signal = "strong"
            signals["gdp"] = f"Strong growth: {gdp:.1f}%"
        elif gdp > 0.0:
            gdp_signal = "moderate"
            signals["gdp"] = f"Moderate growth: {gdp:.1f}%"
        else:
            gdp_signal = "negative"
            signals["gdp"] = f"Negative growth: {gdp:.1f}%"
    else:
        signals["gdp"] = "Data unavailable"

    if unemployment is not None:
        if unemployment < 4.0:
            unemp_signal = "low"
            signals["unemployment"] = f"Low: {unemployment:.1f}%"
        elif unemployment < 5.0:
            unemp_signal = "moderate"
            signals["unemployment"] = f"Moderate: {unemployment:.1f}%"
        elif unemployment < 6.5:
            unemp_signal = "elevated"
            signals["unemployment"] = f"Elevated: {unemployment:.1f}%"
        else:
            unemp_signal = "high"
            signals["unemployment"] = f"High: {unemployment:.1f}%"
    else:
        signals["unemployment"] = "Data unavailable"

    if yield_spread is not None:
        if yield_spread > 1.0:
            spread_signal = "steep"
            signals["yield_spread"] = f"Steep curve: {yield_spread:.2f}%"
        elif yield_spread > 0.0:
            spread_signal = "positive"
            signals["yield_spread"] = f"Positive: {yield_spread:.2f}%"
        elif yield_spread > -0.5:
            spread_signal = "flat"
            signals["yield_spread"] = f"Flat/narrowing: {yield_spread:.2f}%"
        else:
            spread_signal = "inverted"
            signals["yield_spread"] = f"Inverted: {yield_spread:.2f}%"
    else:
        signals["yield_spread"] = "Data unavailable"

    # Classify regime based on combined signals
    # CONTRACTION: GDP negative or (unemployment high + inverted curve)
    if gdp_signal == "negative":
        signals["regime_reason"] = "GDP contraction detected"
        return MacroRegime.CONTRACTION, signals

    if unemp_signal == "high" and spread_signal == "inverted":
        signals["regime_reason"] = "High unemployment with inverted yield curve"
        return MacroRegime.CONTRACTION, signals

    # EXPANSION: Strong GDP + low/moderate unemployment + positive spread
    if gdp_signal == "strong" and unemp_signal in ("low", "moderate"):
        if spread_signal in ("steep", "positive"):
            signals["regime_reason"] = "Strong growth with healthy labor market"
            return MacroRegime.EXPANSION, signals

    # RECOVERY: Moderate GDP + unemployment elevated but improving + spread positive
    if gdp_signal == "moderate" and unemp_signal in ("elevated", "high"):
        signals["regime_reason"] = "Growth resuming with elevated unemployment"
        return MacroRegime.RECOVERY, signals

    if gdp_signal == "moderate" and spread_signal == "steep":
        signals["regime_reason"] = "Moderate growth with steepening curve"
        return MacroRegime.RECOVERY, signals

    # SLOWDOWN: Moderate GDP with flat/narrowing spread, or strong GDP + inverted
    if gdp_signal == "moderate" and spread_signal in ("flat", "inverted"):
        signals["regime_reason"] = "Moderate growth with flattening curve"
        return MacroRegime.SLOWDOWN, signals

    if gdp_signal == "strong" and spread_signal in ("flat", "inverted"):
        signals["regime_reason"] = "Growth persists but yield curve warning"
        return MacroRegime.SLOWDOWN, signals

    if unemp_signal == "elevated" and spread_signal in ("flat", "inverted"):
        signals["regime_reason"] = "Rising unemployment with flat curve"
        return MacroRegime.SLOWDOWN, signals

    # Default: If strong GDP, lean expansion; otherwise slowdown
    if gdp_signal == "strong":
        signals["regime_reason"] = "Strong growth (default classification)"
        return MacroRegime.EXPANSION, signals

    signals["regime_reason"] = "Mixed signals (default classification)"
    return MacroRegime.SLOWDOWN, signals


def detect_regime() -> MacroSnapshot:
    """Detect current macro regime from FRED data.

    Fetches GDP growth, unemployment, CPI, fed funds rate, and yield spread
    from FRED, then classifies the current macro regime.

    Returns:
        MacroSnapshot with regime classification and all indicator values.
    """
    log.info("Fetching FRED macro indicators...")

    gdp = fetch_gdp_growth()
    unemployment = fetch_unemployment()
    cpi = fetch_cpi_yoy()
    fed_funds = fetch_fed_funds_rate()
    yield_spread = fetch_yield_spread_10y2y()

    log.info(
        "FRED data: GDP=%.1f%%, Unemp=%.1f%%, CPI=%.1f%%, FFR=%.2f%%, Spread=%.2f%%",
        gdp or 0.0, unemployment or 0.0, cpi or 0.0,
        fed_funds or 0.0, yield_spread or 0.0,
    )

    regime, signals = _classify_regime(gdp, unemployment, yield_spread)

    if cpi is not None:
        signals["cpi"] = f"CPI YoY: {cpi:.1f}%"
    if fed_funds is not None:
        signals["fed_funds"] = f"Fed Funds: {fed_funds:.2f}%"

    log.info("Detected regime: %s", regime.value)

    return MacroSnapshot(
        regime=regime,
        gdp_growth=gdp,
        unemployment=unemployment,
        cpi_yoy=cpi,
        fed_funds_rate=fed_funds,
        yield_spread_10y2y=yield_spread,
        signals=signals,
    )
