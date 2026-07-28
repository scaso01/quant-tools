"""Google Trends client for search interest analysis.

Uses the pytrends library (unofficial Google Trends API).
Useful for gauging retail interest in stocks, crypto, and macro topics.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("google_trends", threshold=3, recovery_seconds=120.0)


class GoogleTrendsClient:
    """Fetch Google Trends search interest data.

    Requires pytrends: ``pip install pytrends``
    """

    def __init__(self) -> None:
        self._client = None  # lazy-loaded pytrends.TrendReq

    def _ensure_client(self) -> None:
        """Lazy-initialize the pytrends client."""
        if self._client is not None:
            return
        try:
            from pytrends.request import TrendReq

            self._client = TrendReq(hl="en-US", tz=360)
        except ImportError:
            raise DataError(
                "pytrends not installed. pip install pytrends"
            )

    def fetch_interest(
        self,
        keyword: str,
        timeframe: str = "today 12-m",
        geo: str = "",
    ) -> pd.DataFrame:
        """Fetch Google Trends interest over time for a keyword.

        Args:
            keyword: Search term (e.g. "AAPL", "Bitcoin", "inflation").
            timeframe: Time range string. Examples:
                - "today 12-m" (last 12 months, weekly)
                - "today 3-m" (last 3 months, daily)
                - "2020-01-01 2023-12-31" (custom range)
            geo: Country code (e.g. "US", "GB"). Empty for worldwide.

        Returns:
            DataFrame with date index and 'interest' column (0-100 scale).

        Raises:
            DataError: If no data found or library missing.
            NetworkError: If Google Trends is unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "Google Trends circuit breaker is open. Try again later."
            )

        self._ensure_client()

        try:
            self._client.build_payload(
                [keyword],
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop="",
            )
            raw = self._client.interest_over_time()

            if raw is None or raw.empty:
                _breaker.record_success()
                raise DataError(
                    f"No Google Trends data for '{keyword}' "
                    f"(timeframe={timeframe}, geo={geo})"
                )

            # Drop the 'isPartial' column if present
            if "isPartial" in raw.columns:
                raw = raw.drop(columns=["isPartial"])

            # Rename the keyword column to 'interest'
            if keyword in raw.columns:
                raw = raw.rename(columns={keyword: "interest"})
            elif len(raw.columns) == 1:
                raw.columns = ["interest"]

            _breaker.record_success()
            logger.info(
                "Google Trends: '%s' — %d data points (geo=%s)",
                keyword,
                len(raw),
                geo or "worldwide",
            )
            return raw[["interest"]] if "interest" in raw.columns else raw

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            _breaker.record_failure()
            msg = str(exc).lower()
            if any(
                kw in msg
                for kw in ("timeout", "connection", "refused", "network", "429")
            ):
                raise NetworkError(
                    f"Google Trends unreachable: {exc}"
                ) from exc
            raise DataError(
                f"Google Trends fetch failed: {exc}"
            ) from exc

    def fetch_related_queries(self, keyword: str) -> dict:
        """Fetch related queries for a keyword.

        Args:
            keyword: Search term.

        Returns:
            Dict with 'top' and 'rising' keys, each containing a list of
            dicts with 'query' and 'value' keys.

        Raises:
            DataError: If no data found or library missing.
            NetworkError: If Google Trends is unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "Google Trends circuit breaker is open. Try again later."
            )

        self._ensure_client()

        try:
            self._client.build_payload([keyword], cat=0, timeframe="today 12-m")
            raw = self._client.related_queries()

            if not raw or keyword not in raw:
                _breaker.record_success()
                raise DataError(
                    f"No related queries for '{keyword}'"
                )

            result: dict = {"top": [], "rising": []}
            kw_data = raw[keyword]

            for category in ("top", "rising"):
                df = kw_data.get(category)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        result[category].append(
                            {
                                "query": str(row.get("query", "")),
                                "value": int(row.get("value", 0)),
                            }
                        )

            _breaker.record_success()
            logger.info(
                "Google Trends related: '%s' — %d top, %d rising",
                keyword,
                len(result["top"]),
                len(result["rising"]),
            )
            return result

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            _breaker.record_failure()
            msg = str(exc).lower()
            if any(
                kw in msg
                for kw in ("timeout", "connection", "refused", "network", "429")
            ):
                raise NetworkError(
                    f"Google Trends unreachable: {exc}"
                ) from exc
            raise DataError(
                f"Google Trends related queries failed: {exc}"
            ) from exc
