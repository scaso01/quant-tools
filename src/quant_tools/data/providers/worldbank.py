"""World Bank API client for macroeconomic indicators.

Uses the free World Bank API v2 (no auth required).
API: https://api.worldbank.org/v2/country/{code}/indicator/{indicator}?format=json
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import worldbank_breaker

logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldbank.org/v2"

# Common macroeconomic indicators
INDICATORS = {
    "gdp": "NY.GDP.MKTP.CD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "trade_balance": "NE.RSB.GNFS.CD",
    "interest_rate": "FR.INR.RINR",
    "population": "SP.POP.TOTL",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
}


class WorldBankClient:
    """Fetch macroeconomic data from the World Bank API.

    No authentication required. Uses existing httpx dependency.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_indicator(
        self,
        country: str,
        indicator: str,
        date_range: str = "2000:2025",
    ) -> pd.DataFrame:
        """Fetch a World Bank indicator as a DataFrame.

        Args:
            country: ISO 3166-1 alpha-2 country code (e.g. "US", "GB", "CN").
            indicator: Indicator code (e.g. "NY.GDP.MKTP.CD") or shortname
                       from INDICATORS dict (e.g. "gdp", "inflation").
            date_range: Year range in "start:end" format (default "2000:2025").

        Returns:
            DataFrame with columns: year, value, country, indicator.

        Raises:
            DataError: If no data found or API returns error.
            NetworkError: If the API is unreachable.
        """
        if worldbank_breaker.is_open:
            raise NetworkError("World Bank circuit breaker is open. Try again later.")

        # Resolve shortname to indicator code
        indicator_code = INDICATORS.get(indicator.lower(), indicator)

        url = f"{BASE_URL}/country/{country}/indicator/{indicator_code}"
        params = {
            "format": "json",
            "date": date_range,
            "per_page": "500",
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if not isinstance(data, list) or len(data) < 2:
                worldbank_breaker.record_failure()
                raise DataError(
                    f"No World Bank data for {country}/{indicator_code} ({date_range})"
                )

            records = data[1]
            if not records:
                worldbank_breaker.record_failure()
                raise DataError(
                    f"No World Bank data for {country}/{indicator_code} ({date_range})"
                )

            rows = []
            for rec in records:
                value = rec.get("value")
                if value is not None:
                    rows.append({
                        "year": int(rec["date"]),
                        "value": float(value),
                        "country": rec.get("country", {}).get("value", country),
                        "indicator": rec.get("indicator", {}).get("value", indicator_code),
                    })

            if not rows:
                worldbank_breaker.record_failure()
                raise DataError(
                    f"All values null for {country}/{indicator_code} ({date_range})"
                )

            df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
            worldbank_breaker.record_success()
            logger.info(
                "World Bank: %s/%s — %d data points (%d-%d)",
                country, indicator_code, len(df),
                df["year"].min(), df["year"].max(),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            worldbank_breaker.record_failure()
            raise NetworkError(f"World Bank API timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            worldbank_breaker.record_failure()
            raise NetworkError(f"World Bank API error {exc.response.status_code}") from exc
        except Exception as exc:
            worldbank_breaker.record_failure()
            raise DataError(f"World Bank fetch failed: {exc}") from exc

    def fetch_multiple_indicators(
        self,
        country: str,
        indicators: list[str] | None = None,
        date_range: str = "2000:2025",
    ) -> pd.DataFrame:
        """Fetch multiple indicators and pivot into a wide DataFrame.

        Args:
            country: ISO country code.
            indicators: List of indicator names/codes. Defaults to all in INDICATORS.
            date_range: Year range.

        Returns:
            DataFrame with year as index and indicator names as columns.

        Raises:
            DataError: If no data found for any indicator.
        """
        if indicators is None:
            indicators = list(INDICATORS.keys())

        frames = []
        for ind in indicators:
            try:
                df = self.fetch_indicator(country, ind, date_range)
                ind_name = ind if ind in INDICATORS else df["indicator"].iloc[0]
                df = df[["year", "value"]].rename(columns={"value": ind_name})
                frames.append(df)
            except (DataError, NetworkError) as exc:
                logger.warning("Skipping %s for %s: %s", ind, country, exc)

        if not frames:
            raise DataError(f"No World Bank data available for {country}")

        result = frames[0]
        for df in frames[1:]:
            result = result.merge(df, on="year", how="outer")

        return result.sort_values("year").reset_index(drop=True)
