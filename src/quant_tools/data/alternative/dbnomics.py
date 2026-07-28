"""DBnomics client for economic and financial time series.

DBnomics aggregates data from 80+ providers (IMF, OECD, Eurostat, ECB, etc.)
into a single free API. No authentication required.
API: https://api.db.nomics.world/v22/
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("dbnomics", threshold=3, recovery_seconds=90.0)

BASE_URL = "https://api.db.nomics.world/v22"


class DBnomicsClient:
    """Fetch economic time series from DBnomics.

    Uses the free REST API directly via httpx (no library dependency).
    The ``dbnomics`` package (AGPL-3.0) is intentionally avoided for license compliance.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_series(
        self,
        provider: str,
        dataset: str,
        series: str,
    ) -> pd.DataFrame:
        """Fetch a single time series from DBnomics.

        Args:
            provider: Data provider code (e.g. "IMF", "OECD", "Eurostat").
            dataset: Dataset code (e.g. "WEO:2024-04", "MEI").
            series: Series code within the dataset.

        Returns:
            DataFrame with date index and 'value' column.

        Raises:
            DataError: If series not found or empty.
            NetworkError: If API unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "DBnomics circuit breaker is open. Try again later."
            )

        series_id = f"{provider}/{dataset}/{series}"
        url = f"{BASE_URL}/series/{series_id}"
        params = {"observations": "1"}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            series_data = data.get("series", {})
            docs = series_data.get("docs", [])

            if not docs:
                _breaker.record_success()
                raise DataError(
                    f"No DBnomics series found: {series_id}"
                )

            doc = docs[0]
            periods = doc.get("period", [])
            values = doc.get("value", [])

            if not periods or not values:
                _breaker.record_success()
                raise DataError(
                    f"Empty DBnomics series: {series_id}"
                )

            rows = []
            for period, value in zip(periods, values):
                if value is not None and value != "NA":
                    try:
                        rows.append(
                            {
                                "date": str(period),
                                "value": float(value),
                            }
                        )
                    except (ValueError, TypeError):
                        continue

            if not rows:
                _breaker.record_success()
                raise DataError(
                    f"All values null/NA in DBnomics series: {series_id}"
                )

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(
                drop=True
            )
            df = df.set_index("date")

            _breaker.record_success()
            logger.info(
                "DBnomics: %s — %d data points",
                series_id,
                len(df),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"DBnomics API timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"DBnomics API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            _breaker.record_failure()
            raise DataError(
                f"DBnomics fetch failed: {exc}"
            ) from exc

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for datasets/series on DBnomics.

        Args:
            query: Search term (e.g. "US GDP", "inflation").
            limit: Maximum results to return.

        Returns:
            List of dicts with keys: provider_code, dataset_code,
            series_code, name, description.

        Raises:
            DataError: If search fails or no results.
            NetworkError: If API unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "DBnomics circuit breaker is open. Try again later."
            )

        url = f"{BASE_URL}/search"
        params = {"q": query, "limit": str(limit)}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            if not results:
                _breaker.record_success()
                raise DataError(
                    f"No DBnomics results for '{query}'"
                )

            output = []
            for item in results[:limit]:
                output.append(
                    {
                        "provider_code": item.get("provider_code", ""),
                        "dataset_code": item.get("dataset_code", ""),
                        "series_code": item.get("series_code", ""),
                        "name": item.get("series_name", ""),
                        "description": item.get("dataset_name", ""),
                    }
                )

            _breaker.record_success()
            logger.info(
                "DBnomics search: '%s' — %d results",
                query,
                len(output),
            )
            return output

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"DBnomics search timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"DBnomics search error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            _breaker.record_failure()
            raise DataError(
                f"DBnomics search failed: {exc}"
            ) from exc
