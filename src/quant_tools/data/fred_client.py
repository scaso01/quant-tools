"""FRED API wrapper with retry and circuit breaker."""

from __future__ import annotations

import logging

import pandas as pd
from fredapi import Fred

from quant_tools.config import cfg
from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import fred_breaker, retry_with_backoff

log = logging.getLogger("quant_tools")

# FRED series IDs
_GDP_SERIES = "A191RL1Q225SBEA"  # Real GDP growth (annualized)
_UNEMPLOYMENT_SERIES = "UNRATE"  # Unemployment rate
_CPI_SERIES = "CPIAUCSL"  # CPI for all urban consumers
_FED_FUNDS_SERIES = "FEDFUNDS"  # Effective federal funds rate
_YIELD_10Y_SERIES = "DGS10"  # 10-Year Treasury
_YIELD_2Y_SERIES = "DGS2"  # 2-Year Treasury


def _get_fred() -> Fred:
    """Create a Fred client with the configured API key.

    Returns:
        Configured Fred instance.

    Raises:
        DataError: If no FRED API key is configured.
    """
    if not cfg.fred_api_key:
        raise DataError("FRED_API_KEY not configured in .env")
    return Fred(api_key=cfg.fred_api_key)


@retry_with_backoff(max_attempts=3, initial_delay=2.0, api_name="fred")
def fetch_series(series_id: str, start: str | None = None) -> pd.Series:
    """Fetch a FRED data series.

    Args:
        series_id: FRED series identifier (e.g., "UNRATE").
        start: Optional start date (YYYY-MM-DD).

    Returns:
        pandas Series with date index and float values.

    Raises:
        NetworkError: If the FRED API call fails.
        DataError: If no data is returned.
    """
    if fred_breaker.is_open:
        raise NetworkError("FRED circuit breaker is open")

    try:
        fred = _get_fred()
        if start:
            data = fred.get_series(series_id, observation_start=start)
        else:
            data = fred.get_series(series_id)
    except DataError:
        raise
    except Exception as exc:
        fred_breaker.record_failure()
        raise NetworkError(f"FRED fetch failed for {series_id}: {exc}") from exc

    if data is None or data.empty:
        fred_breaker.record_failure()
        raise DataError(f"No FRED data returned for {series_id}")

    fred_breaker.record_success()
    log.debug("Fetched %d observations for %s", len(data), series_id)
    return data


def _latest_value(series_id: str) -> float | None:
    """Fetch the latest non-NaN value from a FRED series.

    Args:
        series_id: FRED series identifier.

    Returns:
        Latest float value, or None on failure.
    """
    try:
        data = fetch_series(series_id)
        clean = data.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])
    except Exception:
        log.warning("Failed to fetch latest value for %s", series_id, exc_info=True)
        return None


def fetch_gdp_growth() -> float | None:
    """Fetch latest real GDP growth rate (annualized).

    Returns:
        GDP growth rate as a percentage, or None on failure.
    """
    return _latest_value(_GDP_SERIES)


def fetch_unemployment() -> float | None:
    """Fetch latest unemployment rate.

    Returns:
        Unemployment rate as a percentage, or None on failure.
    """
    return _latest_value(_UNEMPLOYMENT_SERIES)


def fetch_cpi_yoy() -> float | None:
    """Fetch latest CPI year-over-year change.

    Computes YoY percentage change from the monthly CPI index.

    Returns:
        CPI YoY change as a percentage, or None on failure.
    """
    try:
        data = fetch_series(_CPI_SERIES)
        clean = data.dropna()
        if len(clean) < 13:
            return None
        current = float(clean.iloc[-1])
        year_ago = float(clean.iloc[-13])
        if year_ago == 0:
            return None
        return ((current - year_ago) / year_ago) * 100
    except Exception:
        log.warning("Failed to compute CPI YoY", exc_info=True)
        return None


def fetch_fed_funds_rate() -> float | None:
    """Fetch effective federal funds rate.

    Returns:
        Fed funds rate as a percentage, or None on failure.
    """
    return _latest_value(_FED_FUNDS_SERIES)


def fetch_yield_spread_10y2y() -> float | None:
    """Fetch 10Y-2Y Treasury yield spread.

    Returns:
        Spread in percentage points, or None on failure.
    """
    try:
        y10 = fetch_series(_YIELD_10Y_SERIES)
        y2 = fetch_series(_YIELD_2Y_SERIES)

        y10_clean = y10.dropna()
        y2_clean = y2.dropna()

        if y10_clean.empty or y2_clean.empty:
            return None

        return float(y10_clean.iloc[-1]) - float(y2_clean.iloc[-1])
    except Exception:
        log.warning("Failed to compute 10Y-2Y spread", exc_info=True)
        return None
