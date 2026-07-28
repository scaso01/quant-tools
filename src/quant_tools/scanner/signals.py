"""Signal scoring functions for the stock scanner.

Each function returns a float score in range 0-100.
Higher scores indicate stronger signals.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from quant_tools.data.indicators import bollinger_bands, rsi, sma

log = logging.getLogger("quant_tools")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def momentum_signal(
    close: NDArray[np.float64],
    volume: NDArray[np.float64] | None = None,
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Score based on price momentum.

    Components (equal weight):
    - RSI(14): mapped so 50-70 is bullish (high score), <30 or >80 penalized
    - SMA trend: SMA(20) vs SMA(50) crossover direction
    - 1-month return: positive return scores higher

    Args:
        close: Array of closing prices (at least 60 elements recommended).
        volume: Optional array of volumes for volume-weighted adjustment.
        return_breakdown: If True, return (score, breakdown_dict).

    Returns:
        Momentum score from 0 to 100, or tuple of (score, breakdown) if
        return_breakdown is True.
    """
    if len(close) < 15:
        score = 50.0
        if return_breakdown:
            return score, {"rsi": 50.0, "sma_crossover": 50.0, "return_1m": 50.0, "volume_surge": 0.0}
        return score

    scores: list[float] = []
    breakdown: dict[str, float] = {}

    # 1. RSI component (0-100)
    rsi_values = rsi(close, period=14)
    valid_rsi = rsi_values[~np.isnan(rsi_values)]
    rsi_component = 50.0
    if len(valid_rsi) > 0:
        current_rsi = float(valid_rsi[-1])
        # Ideal RSI: 50-70 range is bullish
        if 50 <= current_rsi <= 70:
            rsi_component = 60 + (current_rsi - 50) * 2  # 60-100
        elif 30 <= current_rsi < 50:
            rsi_component = 30 + (current_rsi - 30) * 1.5  # 30-60
        elif current_rsi < 30:
            rsi_component = current_rsi  # 0-30 (oversold, bearish momentum)
        else:
            # >70: overbought, still has momentum but fading
            rsi_component = max(0, 100 - (current_rsi - 70) * 2)
        rsi_component = _clamp(rsi_component)
        scores.append(rsi_component)
    breakdown["rsi"] = round(rsi_component, 2)

    # 2. SMA trend component (0-100)
    sma_component = 50.0
    if len(close) >= 50:
        sma20 = sma(close, 20)
        sma50 = sma(close, 50)
        valid_mask = ~np.isnan(sma20) & ~np.isnan(sma50)
        if np.any(valid_mask):
            last_sma20 = float(sma20[valid_mask][-1])
            last_sma50 = float(sma50[valid_mask][-1])
            if last_sma50 > 0:
                spread = (last_sma20 - last_sma50) / last_sma50 * 100
                # Spread of +5% = score 100, -5% = score 0
                sma_component = _clamp(50 + spread * 10)
                scores.append(sma_component)
    breakdown["sma_crossover"] = round(sma_component, 2)

    # 3. 1-month return component (0-100)
    return_component = 50.0
    if len(close) >= 21:
        monthly_return = (close[-1] / close[-21] - 1) * 100
        # +10% return = score 100, -10% = score 0
        return_component = _clamp(50 + monthly_return * 5)
        scores.append(return_component)
    breakdown["return_1m"] = round(return_component, 2)

    # 4. Volume surge bonus (if volume provided)
    volume_component = 0.0
    if volume is not None and len(volume) >= 20:
        avg_vol = float(np.mean(volume[-20:]))
        if avg_vol > 0:
            vol_ratio = float(volume[-1]) / avg_vol
            # High volume confirms momentum
            vol_bonus = min(10, (vol_ratio - 1) * 10) if vol_ratio > 1 else 0
            volume_component = round(vol_bonus, 2)
            if scores:
                scores[-1] = _clamp(scores[-1] + vol_bonus)
    breakdown["volume_surge"] = volume_component

    if not scores:
        score = 50.0
        if return_breakdown:
            return score, breakdown
        return score

    score = round(_clamp(float(np.mean(scores))), 2)
    if return_breakdown:
        return score, breakdown
    return score


def value_signal(
    info: dict,
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Score based on valuation metrics.

    Components:
    - PE ratio: lower is better (relative scoring)
    - PB ratio: lower is better

    Args:
        info: Ticker info dict with keys like 'pe_ratio', 'pb_ratio'.
        return_breakdown: If True, return (score, breakdown_dict).

    Returns:
        Value score from 0 to 100, or tuple of (score, breakdown) if
        return_breakdown is True.
    """
    scores: list[float] = []
    breakdown: dict[str, float] = {}

    # PE ratio component
    pe = info.get("pe_ratio")
    pe_component = 50.0
    if pe is not None and isinstance(pe, (int, float)) and pe > 0:
        # PE < 10 = great value (100), PE > 40 = expensive (0)
        pe_component = _clamp(max(0, 100 - (pe - 10) * (100 / 30)))
        scores.append(pe_component)
    breakdown["pe_ratio"] = round(pe_component, 2)

    # PB ratio component
    pb = info.get("pb_ratio")
    pb_component = 50.0
    if pb is not None and isinstance(pb, (int, float)) and pb > 0:
        # PB < 1 = great value (100), PB > 10 = expensive (0)
        pb_component = _clamp(max(0, 100 - (pb - 1) * (100 / 9)))
        scores.append(pb_component)
    breakdown["pb_ratio"] = round(pb_component, 2)

    # Forward PE gives bonus for growth-at-reasonable-price
    forward_pe = info.get("forward_pe")
    forward_pe_bonus = 0.0
    if (
        forward_pe is not None
        and isinstance(forward_pe, (int, float))
        and forward_pe > 0
        and pe is not None
        and isinstance(pe, (int, float))
        and pe > 0
    ):
        # Forward PE lower than trailing = earnings growing
        if forward_pe < pe:
            forward_pe_bonus = min(20, (pe - forward_pe) / pe * 100)
            if scores:
                scores[-1] = _clamp(scores[-1] + forward_pe_bonus)
    breakdown["forward_pe_bonus"] = round(forward_pe_bonus, 2)

    if not scores:
        score = 50.0
        if return_breakdown:
            return score, breakdown
        return score

    score = round(_clamp(float(np.mean(scores))), 2)
    if return_breakdown:
        return score, breakdown
    return score


def quality_signal(
    info: dict,
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Score based on quality metrics.

    Components:
    - ROE: higher is better
    - Profit margin: higher is better
    - Debt/equity: lower is better

    Args:
        info: Ticker info dict with keys like 'roe', 'profit_margin', 'debt_to_equity'.
        return_breakdown: If True, return (score, breakdown_dict).

    Returns:
        Quality score from 0 to 100, or tuple of (score, breakdown) if
        return_breakdown is True.
    """
    scores: list[float] = []
    breakdown: dict[str, float] = {}

    # ROE component (expressed as fraction, e.g., 0.15 = 15%)
    roe = info.get("roe")
    roe_component = 50.0
    if roe is not None and isinstance(roe, (int, float)):
        # ROE > 25% = excellent (100), < 0% = poor (0)
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        roe_component = _clamp(roe_pct * 4)  # 25% ROE = 100
        scores.append(roe_component)
    breakdown["roe"] = round(roe_component, 2)

    # Profit margin component
    margin = info.get("profit_margin")
    margin_component = 50.0
    if margin is not None and isinstance(margin, (int, float)):
        margin_pct = margin * 100 if abs(margin) < 1 else margin
        # 30% margin = score 100, 0% = score 0
        margin_component = _clamp(margin_pct * (100 / 30))
        scores.append(margin_component)
    breakdown["profit_margin"] = round(margin_component, 2)

    # Debt/equity component (lower is better)
    de = info.get("debt_to_equity")
    de_component = 50.0
    if de is not None and isinstance(de, (int, float)) and de >= 0:
        # D/E of 0 = score 100, D/E of 200 = score 0
        de_component = _clamp(max(0, 100 - de * 0.5))
        scores.append(de_component)
    breakdown["debt_equity"] = round(de_component, 2)

    if not scores:
        score = 50.0
        if return_breakdown:
            return score, breakdown
        return score

    score = round(_clamp(float(np.mean(scores))), 2)
    if return_breakdown:
        return score, breakdown
    return score


def mean_reversion_signal(
    close: NDArray[np.float64],
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Score based on mean reversion indicators.

    Higher scores indicate the stock is more likely to revert upward
    (i.e., currently oversold).

    Components:
    - Bollinger %B: how far price is from the bands
    - RSI extremes: oversold (<30) or overbought (>70)

    Args:
        close: Array of closing prices (at least 21 elements recommended).
        return_breakdown: If True, return (score, breakdown_dict).

    Returns:
        Mean reversion score from 0 to 100, or tuple of (score, breakdown) if
        return_breakdown is True.
    """
    if len(close) < 21:
        score = 50.0
        if return_breakdown:
            return score, {"bollinger_pct_b": 50.0, "rsi_extreme": 50.0}
        return score

    scores: list[float] = []
    breakdown: dict[str, float] = {}

    # 1. Bollinger %B component
    bb_component = 50.0
    upper, middle, lower = bollinger_bands(close, period=20, num_std=2.0)
    valid = ~np.isnan(upper) & ~np.isnan(lower)
    if np.any(valid):
        last_upper = float(upper[valid][-1])
        last_lower = float(lower[valid][-1])
        band_width = last_upper - last_lower
        if band_width > 0:
            # %B: 0 = at lower band, 1 = at upper band
            pct_b = (float(close[-1]) - last_lower) / band_width
            # For mean reversion: near lower band = high score (buy signal)
            bb_component = _clamp((1 - pct_b) * 100)
            scores.append(bb_component)
    breakdown["bollinger_pct_b"] = round(bb_component, 2)

    # 2. RSI extreme component
    rsi_component = 50.0
    rsi_values = rsi(close, period=14)
    valid_rsi = rsi_values[~np.isnan(rsi_values)]
    if len(valid_rsi) > 0:
        current_rsi = float(valid_rsi[-1])
        # Oversold (RSI < 30) = high reversion score
        # Overbought (RSI > 70) = low reversion score
        if current_rsi <= 30:
            rsi_component = 100 - current_rsi  # 30->70, 10->90
        elif current_rsi >= 70:
            rsi_component = 100 - current_rsi  # 70->30, 90->10
        else:
            # Neutral zone: score 50
            rsi_component = 50
        rsi_component = _clamp(rsi_component)
        scores.append(rsi_component)
    breakdown["rsi_extreme"] = round(rsi_component, 2)

    if not scores:
        score = 50.0
        if return_breakdown:
            return score, breakdown
        return score

    score = round(_clamp(float(np.mean(scores))), 2)
    if return_breakdown:
        return score, breakdown
    return score


# ── Multi-Asset Signal Functions ─────────────────────────────────────


def _compute_rsi(close: NDArray[np.float64], period: int = 14) -> float:
    """Compute RSI for the latest value."""
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains = float(np.mean(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
    losses = float(np.mean(-deltas[deltas < 0])) if np.any(deltas < 0) else 0.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def crypto_momentum_signal(
    close: NDArray[np.float64],
    volume: NDArray[np.float64] | None = None,
) -> float:
    """Momentum signal tuned for crypto (higher volatility thresholds).

    Similar to momentum_signal but with wider RSI bands and
    stronger volume emphasis for 24/7 markets.
    """
    if len(close) < 30:
        return 50.0

    score = 0.0

    # RSI with wider crypto thresholds
    rsi_val = _compute_rsi(close, 14)
    if rsi_val > 70:
        score += 30  # Strong momentum in crypto
    elif rsi_val > 55:
        score += 20
    elif rsi_val < 25:
        score += 5  # Oversold can still bounce
    else:
        score += 10

    # Short-term trend (7-day vs 21-day SMA)
    if len(close) >= 21:
        sma7 = float(np.mean(close[-7:]))
        sma21 = float(np.mean(close[-21:]))
        spread = (sma7 - sma21) / sma21
        score += _clamp(spread * 500 + 30, 0, 40)

    # Volume surge (crypto markets are volume-sensitive)
    if volume is not None and len(volume) >= 20:
        avg_vol = float(np.mean(volume[-20:]))
        if avg_vol > 0:
            ratio = float(volume[-1]) / avg_vol
            if ratio > 2.0:
                score += 30
            elif ratio > 1.5:
                score += 20
            else:
                score += 10

    return _clamp(score)


def forex_signal(close: NDArray[np.float64]) -> float:
    """Forex trend strength signal via ADX-like calculation.

    Forex markets tend to trend, so we emphasize trend strength.
    """
    if len(close) < 30:
        return 50.0

    score = 0.0

    # Trend direction and strength via moving averages
    sma10 = float(np.mean(close[-10:]))
    sma30 = float(np.mean(close[-30:]))
    trend_strength = abs(sma10 - sma30) / sma30
    score += _clamp(trend_strength * 2000 + 20, 0, 50)

    # Volatility (moderate vol is good for forex trends)
    returns = np.diff(close[-20:]) / close[-20:-1]
    vol = float(np.std(returns) * np.sqrt(252))
    if 0.05 < vol < 0.15:
        score += 30  # Sweet spot
    elif vol < 0.05:
        score += 10  # Too quiet
    else:
        score += 20  # High vol, still tradeable

    # RSI
    rsi_val = _compute_rsi(close, 14)
    if 40 < rsi_val < 60:
        score += 10  # Trending without exhaustion
    elif rsi_val > 70 or rsi_val < 30:
        score += 20  # Strong directional move

    return _clamp(score)


def commodity_signal(
    close: NDArray[np.float64],
    volume: NDArray[np.float64] | None = None,
) -> float:
    """Commodity signal emphasizing mean reversion and seasonal patterns.

    Commodities tend to mean-revert and have seasonal influences.
    """
    if len(close) < 50:
        return 50.0

    score = 0.0

    # Mean reversion: distance from 50-day SMA
    sma50 = float(np.mean(close[-50:]))
    deviation = (float(close[-1]) - sma50) / sma50
    # Score higher when price is far from mean (reversion opportunity)
    score += _clamp(abs(deviation) * 500 + 10, 0, 40)

    # Bollinger Band position
    sma20 = float(np.mean(close[-20:]))
    std20 = float(np.std(close[-20:]))
    if std20 > 0:
        bb_pct = (float(close[-1]) - (sma20 - 2 * std20)) / (4 * std20)
        if bb_pct < 0.2 or bb_pct > 0.8:
            score += 30  # Near bands = opportunity
        else:
            score += 15

    # Volume trend
    if volume is not None and len(volume) >= 20:
        vol_trend = float(np.mean(volume[-5:])) / (float(np.mean(volume[-20:])) + 1e-10)
        if vol_trend > 1.3:
            score += 30
        else:
            score += 15

    return _clamp(score)
