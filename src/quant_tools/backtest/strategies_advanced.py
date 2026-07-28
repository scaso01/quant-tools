"""Advanced strategy signal generators (8 strategies).

All functions return NDArray[np.float64] of positions:
  -1 (short), 0 (neutral), 1 (long).

Follows the same pattern as strategies.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_reversion(
    close: NDArray[np.float64],
    period: int = 20,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
) -> NDArray[np.float64]:
    """Bollinger-based z-score mean reversion strategy.

    Computes a rolling z-score and enters trades when price deviates
    significantly from its moving average.

    - Long when z < -z_entry (price far below mean).
    - Short when z > z_entry (price far above mean).
    - Exit (neutral) when |z| < z_exit.

    Args:
        close: Array of closing prices.
        period: Lookback period for rolling mean/std.
        z_entry: Z-score threshold to enter a position.
        z_exit: Z-score threshold to exit a position.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (neutral/warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    current_pos = 0.0

    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std < 1e-12:
            positions[i] = current_pos
            continue

        z = (close[i] - mean) / std

        if z < -z_entry:
            current_pos = 1.0
        elif z > z_entry:
            current_pos = -1.0
        elif abs(z) < z_exit:
            current_pos = 0.0

        positions[i] = current_pos

    return positions


def pairs_trading(
    close_a: NDArray[np.float64],
    close_b: NDArray[np.float64],
    lookback: int = 60,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
) -> NDArray[np.float64]:
    """Spread-based pairs trading strategy.

    Computes the log-price spread between two assets and trades on
    z-score deviations from the rolling mean.

    - Long spread when z < -z_entry (spread abnormally low).
    - Short spread when z > z_entry (spread abnormally high).
    - Exit when |z| < z_exit.

    The returned positions represent the position in asset A.

    Args:
        close_a: Array of closing prices for asset A.
        close_b: Array of closing prices for asset B.
        lookback: Rolling window for spread mean/std.
        z_entry: Z-score threshold to enter.
        z_exit: Z-score threshold to exit.

    Returns:
        Array of positions for asset A: 1 (long A/short B),
        -1 (short A/long B), 0 (neutral/warmup).
    """
    n = min(len(close_a), len(close_b))
    positions = np.zeros(n, dtype=np.float64)

    # Log spread
    spread = np.log(close_a[:n]) - np.log(close_b[:n])

    current_pos = 0.0
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std < 1e-12:
            positions[i] = current_pos
            continue

        z = (spread[i] - mean) / std

        if z < -z_entry:
            current_pos = 1.0  # Long A, short B
        elif z > z_entry:
            current_pos = -1.0  # Short A, long B
        elif abs(z) < z_exit:
            current_pos = 0.0

        positions[i] = current_pos

    return positions


def trend_following(
    close: NDArray[np.float64],
    fast: int = 20,
    slow: int = 100,
    atr_period: int = 14,
) -> NDArray[np.float64]:
    """Multi-timeframe trend following with ATR trailing stop.

    Enters on fast/slow MA crossover and uses ATR-based trailing stop
    for exit. The stop trails at 2x ATR below the highest high (for longs)
    or above the lowest low (for shorts).

    Args:
        close: Array of closing prices.
        fast: Fast moving average period.
        slow: Slow moving average period.
        atr_period: ATR period for trailing stop.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    warmup = max(slow, atr_period)

    # Compute SMAs
    fast_sma = _rolling_mean(close, fast)
    slow_sma = _rolling_mean(close, slow)

    # Compute ATR from close only (using high-low approximation)
    atr_vals = _close_atr(close, atr_period)

    current_pos = 0.0
    trailing_stop = 0.0
    highest = close[0]
    lowest = close[0]

    for i in range(warmup, n):
        if np.isnan(fast_sma[i]) or np.isnan(slow_sma[i]) or np.isnan(atr_vals[i]):
            positions[i] = 0.0
            continue

        atr_val = atr_vals[i]

        # Check trailing stop exit first
        if current_pos == 1.0 and close[i] < trailing_stop:
            current_pos = 0.0
        elif current_pos == -1.0 and close[i] > trailing_stop:
            current_pos = 0.0

        # Signal generation from MA crossover
        if fast_sma[i] > slow_sma[i] and current_pos <= 0:
            current_pos = 1.0
            highest = close[i]
            trailing_stop = close[i] - 2 * atr_val
        elif fast_sma[i] < slow_sma[i] and current_pos >= 0:
            current_pos = -1.0
            lowest = close[i]
            trailing_stop = close[i] + 2 * atr_val

        # Update trailing stop
        if current_pos == 1.0:
            highest = max(highest, close[i])
            trailing_stop = max(trailing_stop, highest - 2 * atr_val)
        elif current_pos == -1.0:
            lowest = min(lowest, close[i])
            trailing_stop = min(trailing_stop, lowest + 2 * atr_val)

        positions[i] = current_pos

    return positions


def breakout(
    close: NDArray[np.float64],
    volume: NDArray[np.float64] | None = None,
    period: int = 20,
) -> NDArray[np.float64]:
    """Donchian channel breakout strategy.

    Long when price breaks above the period high.
    Short when price breaks below the period low.
    Optional volume confirmation: breakout must occur on above-average volume.

    Args:
        close: Array of closing prices.
        volume: Optional volume array for confirmation.
        period: Lookback period for high/low channels.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (neutral/warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    current_pos = 0.0

    for i in range(period, n):
        window = close[i - period : i]  # Previous 'period' bars (not including current)
        channel_high = np.max(window)
        channel_low = np.min(window)

        # Volume confirmation
        vol_confirmed = True
        if volume is not None and len(volume) == n:
            avg_vol = np.mean(volume[max(0, i - period) : i])
            if avg_vol > 0:
                vol_confirmed = volume[i] > avg_vol

        if close[i] > channel_high and vol_confirmed:
            current_pos = 1.0
        elif close[i] < channel_low and vol_confirmed:
            current_pos = -1.0

        positions[i] = current_pos

    return positions


def factor_rotation(
    close: NDArray[np.float64],
    factor_scores: NDArray[np.float64] | None = None,
    regime: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Factor rotation strategy.

    Rotates between value/momentum/quality based on factor scores or regime.
    - If factor_scores provided: long when score > 0, short when < 0.
    - If regime provided: long in expansion (regime > 0), short in contraction.
    - If neither: uses momentum (60-day return) as default factor.

    Args:
        close: Array of closing prices.
        factor_scores: Optional array of factor scores.
        regime: Optional array of regime indicators.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (warmup/neutral).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    if factor_scores is not None and len(factor_scores) == n:
        for i in range(n):
            if np.isnan(factor_scores[i]):
                positions[i] = 0.0
            elif factor_scores[i] > 0:
                positions[i] = 1.0
            else:
                positions[i] = -1.0
    elif regime is not None and len(regime) == n:
        for i in range(n):
            if np.isnan(regime[i]):
                positions[i] = 0.0
            elif regime[i] > 0:
                positions[i] = 1.0
            else:
                positions[i] = -1.0
    else:
        # Default: momentum factor (60-day return)
        lookback = 60
        for i in range(lookback, n):
            ret = (close[i] / close[i - lookback]) - 1.0
            positions[i] = 1.0 if ret > 0 else -1.0

    return positions


def risk_parity_dynamic(
    close: NDArray[np.float64],
    target_vol: float = 0.15,
) -> NDArray[np.float64]:
    """Dynamic risk parity — scale position size to target portfolio volatility.

    Position sizing: weight = target_vol / realized_vol.
    The position is always long but scaled between 0 and 1.

    Args:
        close: Array of closing prices.
        target_vol: Annualized target volatility (default 0.15 = 15%).

    Returns:
        Array of positions in [0, 1] representing position sizing.
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)
    lookback = 20  # Realized vol lookback

    if n < lookback + 1:
        return positions

    # Compute daily returns
    returns = np.diff(close) / close[:-1]  # length n-1

    daily_target = target_vol / np.sqrt(252)

    for i in range(lookback, n - 1):
        window = returns[i - lookback : i]
        realized_vol = np.std(window, ddof=1)
        if realized_vol < 1e-12:
            positions[i + 1] = 1.0
        else:
            weight = daily_target / realized_vol
            positions[i + 1] = min(max(weight, 0.0), 1.0)

    return positions


def carry_trade(
    close: NDArray[np.float64],
    rate_diff: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Carry trade strategy.

    When the rate differential is positive, go long (earn the carry).
    When negative, go short.

    If rate_diff is not provided, uses the slope of the price curve
    (rolling 20-day return) as a proxy for carry.

    Args:
        close: Array of closing prices.
        rate_diff: Optional array of rate differentials (same length as close).

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (warmup/neutral).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    if rate_diff is not None and len(rate_diff) == n:
        for i in range(n):
            if np.isnan(rate_diff[i]):
                positions[i] = 0.0
            elif rate_diff[i] > 0:
                positions[i] = 1.0
            elif rate_diff[i] < 0:
                positions[i] = -1.0
            else:
                positions[i] = 0.0
    else:
        # Proxy: use rolling 20-day price change as carry signal
        lookback = 20
        for i in range(lookback, n):
            ret = (close[i] / close[i - lookback]) - 1.0
            positions[i] = 1.0 if ret > 0 else -1.0

    return positions


def volatility_targeting(
    close: NDArray[np.float64],
    target_vol: float = 0.15,
    lookback: int = 20,
) -> NDArray[np.float64]:
    """Volatility targeting strategy.

    Always long, but scales position inversely to realized volatility
    to maintain a stable portfolio volatility near target_vol.

    Position = min(target_vol / realized_vol, 1.0)

    This is similar to risk_parity_dynamic but uses the same-period
    scaling convention and exposes lookback as a parameter.

    Args:
        close: Array of closing prices.
        target_vol: Annualized target volatility (default 0.15 = 15%).
        lookback: Lookback period for realized vol calculation.

    Returns:
        Array of positions in [0, 1] representing scaled position sizes.
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    if n < lookback + 1:
        return positions

    returns = np.diff(close) / close[:-1]  # length n-1
    daily_target = target_vol / np.sqrt(252)

    for i in range(lookback, n - 1):
        window = returns[i - lookback : i]
        realized_vol = np.std(window, ddof=1)
        if realized_vol < 1e-12:
            positions[i + 1] = 1.0
        else:
            weight = daily_target / realized_vol
            positions[i + 1] = min(max(weight, 0.0), 1.0)

    return positions


# ── Private helpers ──────────────────────────────────────────────────


def _rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute rolling mean. Returns NaN for indices < period - 1."""
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    cumsum = np.cumsum(arr)
    result[period - 1] = cumsum[period - 1] / period
    for i in range(period, n):
        result[i] = (cumsum[i] - cumsum[i - period]) / period
    return result


def _close_atr(close: np.ndarray, period: int) -> np.ndarray:
    """Approximate ATR from close prices only (no high/low).

    Uses absolute daily price changes as a proxy for true range.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return result

    # Approximate true range from close-to-close
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = 0.0
    for i in range(1, n):
        tr[i] = abs(close[i] - close[i - 1])

    # EMA-style ATR
    if n >= period:
        result[period - 1] = np.mean(tr[1:period + 1]) if period + 1 <= n else np.mean(tr[1:])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result
