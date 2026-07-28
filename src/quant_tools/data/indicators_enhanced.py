"""Enhanced indicator builder using pandas-ta."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("quant_tools")

# Default indicator set with parameters
_DEFAULT_INDICATORS: list[str] = [
    "rsi_14",
    "macd_12_26_9",
    "bbands_20",
    "adx_14",
    "obv",
    "stoch_14_3_3",
    "cci_20",
    "willr_14",
    "atr_14",
]


def build_indicator_set(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with technical indicators appended.

    Uses pandas-ta for indicator computation. Handles empty or short
    DataFrames gracefully by returning them unchanged.

    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
        indicators: List of indicator names. Defaults to a standard set.

    Returns:
        Original DataFrame with indicator columns appended.
    """
    if df is None or df.empty or len(df) < 2:
        return df

    indicator_list = indicators or _DEFAULT_INDICATORS

    try:
        import pandas_ta as ta
    except ImportError:
        log.warning("pandas-ta not installed, returning DataFrame without indicators")
        return df

    result = df.copy()

    for indicator in indicator_list:
        try:
            _apply_indicator(result, indicator, ta)
        except Exception as exc:
            log.debug("Failed to compute indicator %s: %s", indicator, exc)

    return result


def _apply_indicator(
    df: pd.DataFrame, indicator: str, ta,
) -> None:
    """Apply a single indicator to the DataFrame in-place.

    Args:
        df: DataFrame to modify.
        indicator: Indicator name string.
        ta: pandas_ta module reference.
    """
    lower = indicator.lower()

    if lower.startswith("rsi"):
        length = _parse_param(lower, "rsi", 14)
        if len(df) >= length:
            df[f"RSI_{length}"] = ta.rsi(df["Close"], length=length)

    elif lower.startswith("macd"):
        parts = lower.replace("macd_", "").split("_") if "_" in lower else []
        fast = int(parts[0]) if len(parts) >= 1 else 12
        slow = int(parts[1]) if len(parts) >= 2 else 26
        signal = int(parts[2]) if len(parts) >= 3 else 9
        if len(df) >= slow:
            macd_df = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
            if macd_df is not None:
                for col in macd_df.columns:
                    df[col] = macd_df[col]

    elif lower.startswith("bbands") or lower.startswith("bb"):
        length = _parse_param(lower, "bbands", 20)
        if length == 20:
            length = _parse_param(lower, "bb", 20)
        if len(df) >= length:
            bb_df = ta.bbands(df["Close"], length=length)
            if bb_df is not None:
                for col in bb_df.columns:
                    df[col] = bb_df[col]

    elif lower.startswith("adx"):
        length = _parse_param(lower, "adx", 14)
        if len(df) >= length * 2:
            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=length)
            if adx_df is not None:
                for col in adx_df.columns:
                    df[col] = adx_df[col]

    elif lower == "obv":
        df["OBV"] = ta.obv(df["Close"], df["Volume"])

    elif lower.startswith("stoch"):
        parts = lower.replace("stoch_", "").split("_") if "_" in lower else []
        k = int(parts[0]) if len(parts) >= 1 else 14
        d = int(parts[1]) if len(parts) >= 2 else 3
        smooth_k = int(parts[2]) if len(parts) >= 3 else 3
        if len(df) >= k:
            stoch_df = ta.stoch(
                df["High"], df["Low"], df["Close"],
                k=k, d=d, smooth_k=smooth_k,
            )
            if stoch_df is not None:
                for col in stoch_df.columns:
                    df[col] = stoch_df[col]

    elif lower.startswith("cci"):
        length = _parse_param(lower, "cci", 20)
        if len(df) >= length:
            df[f"CCI_{length}"] = ta.cci(
                df["High"], df["Low"], df["Close"], length=length,
            )

    elif lower.startswith("willr"):
        length = _parse_param(lower, "willr", 14)
        if len(df) >= length:
            df[f"WILLR_{length}"] = ta.willr(
                df["High"], df["Low"], df["Close"], length=length,
            )

    elif lower.startswith("atr"):
        length = _parse_param(lower, "atr", 14)
        if len(df) >= length:
            df[f"ATR_{length}"] = ta.atr(
                df["High"], df["Low"], df["Close"], length=length,
            )


def _parse_param(indicator_str: str, prefix: str, default: int) -> int:
    """Parse a numeric parameter from an indicator string.

    "rsi_14" -> 14, "rsi" -> default

    Args:
        indicator_str: Full indicator string.
        prefix: Indicator prefix to strip.
        default: Default value if no parameter found.

    Returns:
        Parsed integer parameter.
    """
    suffix = indicator_str.replace(f"{prefix}_", "").replace(prefix, "")
    if suffix and suffix.isdigit():
        return int(suffix)
    return default
