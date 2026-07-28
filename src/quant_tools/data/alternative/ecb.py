"""ECB exchange rate client for foreign exchange data.

Uses the ECB Statistical Data Warehouse REST API (free, no auth).
Provides daily EUR-based exchange rates since 1999.
API: https://data-api.ecb.europa.eu/service/data/EXR/
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("ecb", threshold=3, recovery_seconds=90.0)

BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR"


class ECBClient:
    """Fetch exchange rates from the European Central Bank.

    No authentication required. EUR is always the base currency.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_exchange_rates(
        self,
        currency: str = "USD",
        start: str = "1999-01-04",
    ) -> pd.DataFrame:
        """Fetch daily ECB exchange rates (EUR-based).

        Args:
            currency: Target currency ISO code (e.g. "USD", "GBP", "JPY").
            start: Start date as YYYY-MM-DD (ECB data begins 1999-01-04).

        Returns:
            DataFrame with columns: date, rate (EUR per unit of currency,
            or units of currency per EUR depending on ECB convention).

        Raises:
            DataError: If no exchange rate data found.
            NetworkError: If ECB API unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "ECB circuit breaker is open. Try again later."
            )

        # ECB SDMX REST endpoint for exchange rates
        # Format: D.{currency}.EUR.SP00.A
        # D=Daily, SP00=spot, A=average
        key = f"D.{currency.upper()}.EUR.SP00.A"
        url = f"{BASE_URL}/{key}"
        params = {
            "startPeriod": start,
            "format": "csvdata",
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                content = resp.text

            if not content or len(content.strip()) < 10:
                _breaker.record_success()
                raise DataError(
                    f"No ECB exchange rate data for {currency} "
                    f"(from {start})"
                )

            # Parse CSV response
            from io import StringIO

            raw_df = pd.read_csv(StringIO(content))

            # ECB CSV has TIME_PERIOD and OBS_VALUE columns
            if "TIME_PERIOD" not in raw_df.columns or "OBS_VALUE" not in raw_df.columns:
                _breaker.record_success()
                raise DataError(
                    f"Unexpected ECB response format for {currency}"
                )

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw_df["TIME_PERIOD"]),
                    "rate": pd.to_numeric(
                        raw_df["OBS_VALUE"], errors="coerce"
                    ),
                }
            )
            df = df.dropna(subset=["rate"]).sort_values("date").reset_index(
                drop=True
            )

            if df.empty:
                _breaker.record_success()
                raise DataError(
                    f"All values null for ECB {currency} rates"
                )

            _breaker.record_success()
            logger.info(
                "ECB: EUR/%s — %d data points (%s to %s)",
                currency,
                len(df),
                df["date"].iloc[0].strftime("%Y-%m-%d"),
                df["date"].iloc[-1].strftime("%Y-%m-%d"),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"ECB API timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"ECB API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            _breaker.record_failure()
            raise DataError(
                f"ECB exchange rate fetch failed: {exc}"
            ) from exc
