"""Cross-asset correlation analysis with rolling windows."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quant_tools.models import AssetClass, CrossAssetCorrelation, RollingCorrelation


def calculate_cross_asset_correlations(
    symbols: list[str],
    period: str = "1y",
    asset_classes: list[AssetClass] | None = None,
) -> CrossAssetCorrelation:
    """Compute full pairwise correlation matrix across asset classes.

    Fetches all symbols via ProviderRegistry, aligns by date intersection
    (inner join -- handles different trading calendars: equities 252d/yr,
    crypto 365d/yr, forex ~260d/yr).

    Args:
        symbols: List of symbols to correlate.
        period: Data period for yfinance/providers.
        asset_classes: Optional list of asset classes matching symbols order.

    Returns:
        CrossAssetCorrelation with full correlation matrix and metadata.
    """
    from quant_tools.data.providers.registry import ProviderRegistry

    registry = ProviderRegistry()

    # Fetch returns for each symbol
    returns_dict: dict[str, pd.Series] = {}
    detected_classes: list[AssetClass] = []

    for i, symbol in enumerate(symbols):
        ac = asset_classes[i] if asset_classes else registry.detect_asset_class(symbol)
        detected_classes.append(ac)
        provider = registry.get_provider(ac)
        try:
            df = provider.fetch_ohlcv(symbol, period=period)
            returns_dict[symbol] = df["Close"].pct_change().dropna()
        except Exception:
            continue

    if len(returns_dict) < 2:
        return CrossAssetCorrelation(
            symbols=list(returns_dict.keys()),
            asset_classes=detected_classes[: len(returns_dict)],
        )

    # Align by date intersection (inner join)
    returns_df = pd.DataFrame(returns_dict).dropna()

    # Compute correlation matrix
    corr_matrix = returns_df.corr().values

    return CrossAssetCorrelation(
        symbols=list(returns_df.columns),
        asset_classes=detected_classes[: len(returns_df.columns)],
        correlation_matrix=corr_matrix,
    )


def calculate_rolling_correlations(
    returns_a: NDArray[np.float64],
    returns_b: NDArray[np.float64],
    window: int = 63,
) -> tuple[list[float], list[int]]:
    """Compute rolling pairwise correlation.

    Args:
        returns_a: Array of returns for asset A.
        returns_b: Array of returns for asset B.
        window: Rolling window size in periods (default 63 = ~3 months).

    Returns:
        Tuple of (correlations, indices) where indices are the end positions
        of each window.
    """
    n = min(len(returns_a), len(returns_b))
    if n < window:
        return [], []

    a, b = returns_a[:n], returns_b[:n]
    correlations = []
    indices = []

    for i in range(window, n + 1):
        window_a = a[i - window : i]
        window_b = b[i - window : i]
        corr = np.corrcoef(window_a, window_b)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations.append(float(corr))
        indices.append(i - 1)

    return correlations, indices
