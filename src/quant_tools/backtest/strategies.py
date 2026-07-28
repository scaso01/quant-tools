"""Strategy signal generators. Each returns a numpy array of positions (-1, 0, 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_tools.data.indicators import bollinger_bands, rsi, sma


def sma_crossover(
    close: NDArray[np.float64], fast: int = 20, slow: int = 50
) -> NDArray[np.float64]:
    """SMA crossover strategy: long when fast SMA > slow SMA, short otherwise.

    Args:
        close: Array of closing prices.
        fast: Fast SMA period.
        slow: Slow SMA period.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (no position during warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)

    for i in range(n):
        if np.isnan(fast_sma[i]) or np.isnan(slow_sma[i]):
            positions[i] = 0.0
        elif fast_sma[i] > slow_sma[i]:
            positions[i] = 1.0
        else:
            positions[i] = -1.0

    return positions


def bollinger_strategy(
    close: NDArray[np.float64], period: int = 20, num_std: float = 2.0
) -> NDArray[np.float64]:
    """Bollinger Bands mean-reversion strategy.

    Buy (1) when price crosses below the lower band.
    Sell (-1) when price crosses above the upper band.
    Hold previous position otherwise.

    Args:
        close: Array of closing prices.
        period: Bollinger Bands lookback period.
        num_std: Number of standard deviations for bands.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (neutral/warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    upper, _middle, lower = bollinger_bands(close, period, num_std)

    current_pos = 0.0
    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            positions[i] = 0.0
            continue

        if close[i] <= lower[i]:
            current_pos = 1.0
        elif close[i] >= upper[i]:
            current_pos = -1.0

        positions[i] = current_pos

    return positions


def rsi_strategy(
    close: NDArray[np.float64],
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> NDArray[np.float64]:
    """RSI strategy: buy on oversold, sell on overbought.

    Args:
        close: Array of closing prices.
        period: RSI lookback period.
        oversold: RSI threshold to go long.
        overbought: RSI threshold to go short.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (neutral/warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    rsi_values = rsi(close, period)

    current_pos = 0.0
    for i in range(n):
        if np.isnan(rsi_values[i]):
            positions[i] = 0.0
            continue

        if rsi_values[i] <= oversold:
            current_pos = 1.0
        elif rsi_values[i] >= overbought:
            current_pos = -1.0

        positions[i] = current_pos

    return positions


def momentum_strategy(
    close: NDArray[np.float64], lookback: int = 60, threshold: float = 0.0
) -> NDArray[np.float64]:
    """Momentum strategy: long if past return exceeds threshold.

    Args:
        close: Array of closing prices.
        lookback: Number of periods to measure momentum.
        threshold: Minimum return to go long (default 0.0).

    Returns:
        Array of positions: 1 (long), -1 (short), 0 (warmup).
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    for i in range(lookback, n):
        ret = (close[i] / close[i - lookback]) - 1.0
        if ret > threshold:
            positions[i] = 1.0
        else:
            positions[i] = -1.0

    return positions


def factor_strategy(
    close: NDArray[np.float64],
    scores: NDArray[np.float64],
    threshold: float = 0.5,
) -> NDArray[np.float64]:
    """Factor-based strategy: long if factor score exceeds threshold.

    Args:
        close: Array of closing prices (used for alignment).
        scores: Array of factor scores (same length as close).
        threshold: Minimum score to go long.

    Returns:
        Array of positions: 1 (long), -1 (short), 0 if score is NaN.
    """
    n = len(close)
    positions = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i >= len(scores) or np.isnan(scores[i]):
            positions[i] = 0.0
        elif scores[i] > threshold:
            positions[i] = 1.0
        else:
            positions[i] = -1.0

    return positions
