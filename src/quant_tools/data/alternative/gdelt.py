"""GDELT event database client for geopolitical event monitoring.

Uses the gdeltPyR library to query the GDELT 2.0 event database.
GDELT monitors broadcast, print, and web news from nearly every country
and translates events into 300+ categories.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("gdelt", threshold=3, recovery_seconds=90.0)


class GdeltClient:
    """Fetch geopolitical events and sentiment from the GDELT database.

    Requires gdeltPyR: ``pip install gdeltPyR``
    """

    def __init__(self) -> None:
        self._gd = None  # lazy-loaded gdeltPyR.gdelt instance

    def _ensure_client(self) -> None:
        """Lazy-initialize the gdeltPyR client."""
        if self._gd is not None:
            return
        try:
            from gdeltPyR import gdelt as gdelt_module

            self._gd = gdelt_module(version=2)
        except ImportError:
            raise DataError(
                "gdeltPyR not installed. pip install gdeltPyR"
            )

    def fetch_events(
        self,
        query: str,
        start_date: str = "",
        end_date: str = "",
        max_records: int = 250,
    ) -> pd.DataFrame:
        """Fetch GDELT events matching a query.

        Args:
            query: Search term (e.g. ticker, company name, country).
            start_date: Start date as YYYY-MM-DD. Defaults to 7 days ago.
            end_date: End date as YYYY-MM-DD. Defaults to today.
            max_records: Maximum number of records to return.

        Returns:
            DataFrame with columns: date, title, url, tone, source_country.

        Raises:
            DataError: If no events found or library missing.
            NetworkError: If GDELT API is unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError("GDELT circuit breaker is open. Try again later.")

        self._ensure_client()

        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            # gdeltPyR expects dates as 'YYYY Mon DD' or similar
            start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime(
                "%Y %b %d"
            )
            end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y %b %d")

            raw = self._gd.Search(
                [start_fmt, end_fmt],
                table="events",
                coverage=False,
            )

            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                _breaker.record_success()
                raise DataError(f"No GDELT events found for '{query}'")

            # Filter by query term in ActionGeo_FullName or SOURCEURL
            mask = pd.Series(False, index=raw.index)
            for col in ["ActionGeo_FullName", "SOURCEURL", "Actor1Name"]:
                if col in raw.columns:
                    mask = mask | raw[col].astype(str).str.contains(
                        query, case=False, na=False
                    )
            filtered = raw[mask].head(max_records)

            if filtered.empty:
                _breaker.record_success()
                raise DataError(f"No GDELT events matching '{query}'")

            rows = []
            for _, row in filtered.iterrows():
                rows.append(
                    {
                        "date": str(row.get("SQLDATE", "")),
                        "title": str(row.get("SOURCEURL", "")),
                        "url": str(row.get("SOURCEURL", "")),
                        "tone": float(row.get("AvgTone", 0.0)),
                        "source_country": str(
                            row.get("ActionGeo_CountryCode", "")
                        ),
                    }
                )

            _breaker.record_success()
            df = pd.DataFrame(rows)
            logger.info(
                "GDELT: '%s' — %d events (%s to %s)",
                query,
                len(df),
                start_date,
                end_date,
            )
            return df

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            _breaker.record_failure()
            msg = str(exc).lower()
            if any(
                kw in msg
                for kw in ("timeout", "connection", "refused", "network")
            ):
                raise NetworkError(f"GDELT API unreachable: {exc}") from exc
            raise DataError(f"GDELT fetch failed: {exc}") from exc

    def fetch_sentiment_timeline(
        self,
        query: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch average sentiment (tone) over time for a query.

        Args:
            query: Search term.
            days: Number of days to look back.

        Returns:
            DataFrame with columns: date, avg_tone, article_count.

        Raises:
            DataError: If no data found or library missing.
            NetworkError: If GDELT API is unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError("GDELT circuit breaker is open. Try again later.")

        self._ensure_client()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime(
                "%Y %b %d"
            )
            end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y %b %d")

            raw = self._gd.Search(
                [start_fmt, end_fmt],
                table="events",
                coverage=False,
            )

            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                _breaker.record_success()
                raise DataError(
                    f"No GDELT sentiment data for '{query}' ({days}d)"
                )

            # Filter by query
            mask = pd.Series(False, index=raw.index)
            for col in ["ActionGeo_FullName", "SOURCEURL", "Actor1Name"]:
                if col in raw.columns:
                    mask = mask | raw[col].astype(str).str.contains(
                        query, case=False, na=False
                    )
            filtered = raw[mask]

            if filtered.empty:
                _breaker.record_success()
                raise DataError(
                    f"No GDELT sentiment data matching '{query}'"
                )

            # Aggregate by date
            filtered = filtered.copy()
            filtered["_date"] = pd.to_datetime(
                filtered["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
            )
            daily = (
                filtered.groupby("_date")
                .agg(avg_tone=("AvgTone", "mean"), article_count=("AvgTone", "count"))
                .reset_index()
                .rename(columns={"_date": "date"})
                .sort_values("date")
                .reset_index(drop=True)
            )

            _breaker.record_success()
            logger.info(
                "GDELT sentiment: '%s' — %d days of data",
                query,
                len(daily),
            )
            return daily

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            _breaker.record_failure()
            msg = str(exc).lower()
            if any(
                kw in msg
                for kw in ("timeout", "connection", "refused", "network")
            ):
                raise NetworkError(f"GDELT API unreachable: {exc}") from exc
            raise DataError(
                f"GDELT sentiment fetch failed: {exc}"
            ) from exc
