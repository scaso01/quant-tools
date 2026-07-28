"""Technical indicators implemented in pure numpy (no ta-lib dependency)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_tools.errors import ComputeError


def sma(prices: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple Moving Average.

    Args:
        prices: Array of prices.
        period: Lookback period.

    Returns:
        Array of SMA values (NaN for first period-1 elements).
    """
    if period < 1:
        raise ComputeError(f"SMA period must be >= 1, got {period}")
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    result = np.full_like(prices, np.nan, dtype=np.float64)
    cumsum = np.cumsum(prices)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return result


def ema(prices: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Exponential Moving Average.

    Args:
        prices: Array of prices.
        period: Lookback period.

    Returns:
        Array of EMA values (NaN for first period-1 elements).
    """
    if period < 1:
        raise ComputeError(f"EMA period must be >= 1, got {period}")
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=np.float64)

    result = np.full_like(prices, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)

    # Seed with SMA
    result[period - 1] = np.mean(prices[:period])

    for i in range(period, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

    return result


def rsi(prices: NDArray[np.float64], period: int = 14) -> NDArray[np.float64]:
    """Relative Strength Index.

    Args:
        prices: Array of closing prices.
        period: RSI lookback period (default 14).

    Returns:
        Array of RSI values (0-100, NaN for first period elements).
    """
    if period < 1:
        raise ComputeError(f"RSI period must be >= 1, got {period}")
    if len(prices) < period + 1:
        return np.full_like(prices, np.nan, dtype=np.float64)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(len(prices), np.nan, dtype=np.float64)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def bollinger_bands(
    prices: NDArray[np.float64], period: int = 20, num_std: float = 2.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bollinger Bands.

    Args:
        prices: Array of closing prices.
        period: SMA lookback period (default 20).
        num_std: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band) arrays.
    """
    if period < 1:
        raise ComputeError(f"Bollinger period must be >= 1, got {period}")

    middle = sma(prices, period)

    std = np.full_like(prices, np.nan, dtype=np.float64)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1: i + 1], ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std

    return upper, middle, lower


def macd(
    prices: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Moving Average Convergence Divergence.

    Args:
        prices: Array of closing prices.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) arrays.
    """
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)

    macd_line = fast_ema - slow_ema

    # Signal line: EMA of MACD line (only where MACD is valid)
    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full_like(prices, np.nan, dtype=np.float64)

    valid_macd = macd_line[valid_mask]
    if len(valid_macd) >= signal_period:
        signal_values = ema(valid_macd, signal_period)
        signal_line[valid_mask] = signal_values

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    """Average True Range.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        period: ATR lookback period (default 14).

    Returns:
        Array of ATR values.
    """
    if period < 1:
        raise ComputeError(f"ATR period must be >= 1, got {period}")

    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    result = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        result[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def volume_ratio(volumes: NDArray[np.float64], period: int = 20) -> NDArray[np.float64]:
    """Volume ratio: current volume / average volume.

    Args:
        volumes: Array of volume values.
        period: Lookback period for average (default 20).

    Returns:
        Array of volume ratios.
    """
    avg_vol = sma(volumes.astype(np.float64), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(avg_vol > 0, volumes / avg_vol, np.nan)
    return ratio
