"""Factor z-score ranking for stock screening."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import zscore

from quant_tools.data.market_data import fetch_prices, fetch_ticker_info
from quant_tools.errors import DataError
from quant_tools.models import FactorScore

log = logging.getLogger("quant_tools")

# Default factor weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 0.33,
    "momentum": 0.33,
    "quality": 0.34,
}


def _safe_zscore(values: list[float | None]) -> np.ndarray:
    """Compute z-scores, replacing NaN with 0.

    Args:
        values: List of raw metric values (may contain None).

    Returns:
        numpy array of z-scores with NaN replaced by 0.
    """
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)

    if len(arr) < 2 or np.all(np.isnan(arr)):
        return np.zeros(len(arr))

    # Replace NaN with median for z-score calculation, then zero out NaN positions
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros(len(arr))

    valid = arr[~nan_mask]
    if np.std(valid) == 0:
        return np.zeros(len(arr))

    median_val = np.nanmedian(arr)
    filled = np.where(nan_mask, median_val, arr)
    z = zscore(filled)
    z[nan_mask] = 0.0

    return z


def _compute_value_scores(infos: list[dict[str, object]]) -> np.ndarray:
    """Compute value z-scores from PE, PB, dividend yield.

    Lower PE/PB = higher value. Higher dividend yield = higher value.

    Args:
        infos: List of ticker info dicts.

    Returns:
        numpy array of value z-scores.
    """
    pe_values = [info.get("pe_ratio") for info in infos]
    pb_values = [info.get("pb_ratio") for info in infos]
    div_values = [info.get("dividend_yield") for info in infos]

    # Invert PE and PB (lower is better, so negate z-scores)
    pe_z = -_safe_zscore(pe_values)
    pb_z = -_safe_zscore(pb_values)
    # Higher dividend yield is better
    div_z = _safe_zscore(div_values)

    return (pe_z + pb_z + div_z) / 3


def _compute_momentum_scores(symbols: list[str]) -> np.ndarray:
    """Compute momentum z-scores from 6-month and 12-month returns.

    Args:
        symbols: List of ticker symbols.

    Returns:
        numpy array of momentum z-scores.
    """
    returns_6m: list[float | None] = []
    returns_12m: list[float | None] = []

    for symbol in symbols:
        try:
            prices = fetch_prices(symbol, period="1y")
            if prices is None or len(prices) < 20:
                returns_6m.append(None)
                returns_12m.append(None)
                continue

            close = prices["Close"].values
            current = float(close[-1])

            # 6-month return (~126 trading days)
            idx_6m = min(126, len(close) - 1)
            price_6m = float(close[-idx_6m - 1]) if idx_6m < len(close) else float(close[0])
            ret_6m = (current - price_6m) / price_6m if price_6m != 0 else 0.0
            returns_6m.append(ret_6m)

            # 12-month return (~252 trading days)
            price_12m = float(close[0])
            ret_12m = (current - price_12m) / price_12m if price_12m != 0 else 0.0
            returns_12m.append(ret_12m)

        except Exception:
            log.debug("Failed to compute momentum for %s", symbol, exc_info=True)
            returns_6m.append(None)
            returns_12m.append(None)

    z_6m = _safe_zscore(returns_6m)
    z_12m = _safe_zscore(returns_12m)

    return (z_6m + z_12m) / 2


def _compute_quality_scores(infos: list[dict[str, object]]) -> np.ndarray:
    """Compute quality z-scores from ROE, profit margin, debt/equity.

    Higher ROE = higher quality. Higher margin = higher quality.
    Lower debt/equity = higher quality.

    Args:
        infos: List of ticker info dicts.

    Returns:
        numpy array of quality z-scores.
    """
    roe_values = [info.get("roe") for info in infos]
    margin_values = [info.get("profit_margin") for info in infos]
    debt_values = [info.get("debt_to_equity") for info in infos]

    roe_z = _safe_zscore(roe_values)
    margin_z = _safe_zscore(margin_values)
    # Lower debt is better, so negate
    debt_z = -_safe_zscore(debt_values)

    return (roe_z + margin_z + debt_z) / 3


def calculate_factor_scores(
    symbols: list[str],
    weights: dict[str, float] | None = None,
) -> list[FactorScore]:
    """Calculate composite factor scores for a list of symbols.

    Args:
        symbols: List of ticker symbols to score.
        weights: Optional dict mapping factor names to weights.
            Defaults to equal weighting: value=0.33, momentum=0.33, quality=0.34.

    Returns:
        List of FactorScore objects sorted by composite_z (descending).

    Raises:
        DataError: If no valid symbols can be scored.
    """
    if not symbols:
        raise DataError("No symbols provided for factor scoring")

    w = weights or DEFAULT_WEIGHTS

    # Fetch ticker info for all symbols
    infos: list[dict[str, object]] = []
    valid_symbols: list[str] = []

    for symbol in symbols:
        try:
            info = fetch_ticker_info(symbol)
            infos.append(info)
            valid_symbols.append(symbol)
        except Exception:
            log.warning("Skipping %s: failed to fetch info", symbol)

    if not valid_symbols:
        raise DataError("No valid symbols could be fetched for factor scoring")

    log.info("Computing factor scores for %d symbols", len(valid_symbols))

    # Compute factor z-scores
    value_z = _compute_value_scores(infos)
    momentum_z = _compute_momentum_scores(valid_symbols)
    quality_z = _compute_quality_scores(infos)

    # Composite z-score
    composite = (
        w.get("value", 0.33) * value_z
        + w.get("momentum", 0.33) * momentum_z
        + w.get("quality", 0.34) * quality_z
    )

    # Build FactorScore objects
    results: list[FactorScore] = []
    for i, symbol in enumerate(valid_symbols):
        info = infos[i]
        results.append(FactorScore(
            symbol=symbol,
            name=str(info.get("name", symbol)),
            value_z=float(value_z[i]),
            momentum_z=float(momentum_z[i]),
            quality_z=float(quality_z[i]),
            composite_z=float(composite[i]),
            sector=str(info.get("sector", "")),
        ))

    # Sort by composite z-score descending, assign ranks
    results.sort(key=lambda fs: fs.composite_z, reverse=True)
    for rank, fs in enumerate(results, start=1):
        fs.rank = rank

    return results
