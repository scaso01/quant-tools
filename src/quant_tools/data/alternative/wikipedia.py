"""Wikipedia pageviews client for tracking article popularity.

Uses the free Wikimedia REST API (no auth required).
Useful for gauging public interest in companies, events, and topics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("wikipedia", threshold=3, recovery_seconds=60.0)

BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


class WikipediaPageviews:
    """Fetch daily pageview counts for Wikipedia articles.

    No authentication required. Uses the free Wikimedia REST API.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_pageviews(
        self,
        article: str,
        start: str = "",
        end: str = "",
        project: str = "en.wikipedia",
    ) -> pd.DataFrame:
        """Fetch daily pageviews for a Wikipedia article.

        Args:
            article: Article title (e.g. "Apple_Inc.", "Bitcoin").
                Spaces should be replaced with underscores.
            start: Start date as YYYY-MM-DD. Defaults to 90 days ago.
            end: End date as YYYY-MM-DD. Defaults to yesterday.
            project: Wikimedia project (default "en.wikipedia").

        Returns:
            DataFrame with columns: date, views.

        Raises:
            DataError: If no pageview data found.
            NetworkError: If API unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "Wikipedia circuit breaker is open. Try again later."
            )

        # Default date range: last 90 days
        if not start:
            start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        if not end:
            end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Wikimedia API expects YYYYMMDD format
        start_fmt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
        end_fmt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")

        # Replace spaces with underscores in article title
        article_clean = article.replace(" ", "_")

        url = (
            f"{BASE_URL}/{project}/all-access/all-agents/"
            f"{article_clean}/daily/{start_fmt}/{end_fmt}"
        )

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(
                    url,
                    headers={"User-Agent": "quant-tools/0.1 pageview-tracker"},
                )
                resp.raise_for_status()
                data = resp.json()

            items = data.get("items", [])
            if not items:
                _breaker.record_success()
                raise DataError(
                    f"No Wikipedia pageviews for '{article}' ({start} to {end})"
                )

            rows = []
            for item in items:
                date_str = item.get("timestamp", "")
                if len(date_str) >= 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                rows.append(
                    {
                        "date": date_str,
                        "views": int(item.get("views", 0)),
                    }
                )

            _breaker.record_success()
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            logger.info(
                "Wikipedia: '%s' — %d days, avg %.0f views/day",
                article,
                len(df),
                df["views"].mean(),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"Wikipedia API timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"Wikipedia API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            _breaker.record_failure()
            raise DataError(
                f"Wikipedia pageviews fetch failed: {exc}"
            ) from exc
