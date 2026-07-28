"""CoinGecko client for cryptocurrency market data.

Uses the free CoinGecko API v3 (no auth required for basic endpoints).
Provides market data, trending coins, and category breakdowns.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import coingecko_breaker

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoClient:
    """Fetch cryptocurrency market data from CoinGecko.

    Uses the free public API (rate-limited to ~10-30 calls/min).
    Reuses the existing ``coingecko_breaker`` from retry.py.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_market_data(
        self,
        coin_id: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical market data for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin identifier (e.g. "bitcoin", "ethereum").
            days: Number of days of history (1, 7, 14, 30, 90, 180, 365, max).

        Returns:
            DataFrame with columns: date, price, market_cap, volume.

        Raises:
            DataError: If no data returned.
            NetworkError: If CoinGecko unreachable or circuit breaker open.
        """
        if coingecko_breaker.is_open:
            raise NetworkError(
                "CoinGecko circuit breaker is open. Try again later."
            )

        url = f"{BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days)}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            prices = data.get("prices", [])
            market_caps = data.get("market_caps", [])
            volumes = data.get("total_volumes", [])

            if not prices:
                coingecko_breaker.record_success()
                raise DataError(
                    f"No CoinGecko market data for '{coin_id}' ({days}d)"
                )

            rows = []
            for i, (ts, price) in enumerate(prices):
                row = {
                    "date": datetime.fromtimestamp(ts / 1000, tz=UTC).strftime(
                        "%Y-%m-%d"
                    ),
                    "price": float(price),
                    "market_cap": float(market_caps[i][1])
                    if i < len(market_caps)
                    else 0.0,
                    "volume": float(volumes[i][1])
                    if i < len(volumes)
                    else 0.0,
                }
                rows.append(row)

            coingecko_breaker.record_success()
            df = pd.DataFrame(rows)
            # Deduplicate by date (intraday data may have multiple entries)
            df = df.drop_duplicates(subset=["date"], keep="last").reset_index(
                drop=True
            )
            logger.info(
                "CoinGecko: '%s' — %d data points (%dd)",
                coin_id,
                len(df),
                days,
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            coingecko_breaker.record_failure()
            raise NetworkError(
                f"CoinGecko API timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            coingecko_breaker.record_failure()
            raise NetworkError(
                f"CoinGecko API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            coingecko_breaker.record_failure()
            raise DataError(
                f"CoinGecko market data fetch failed: {exc}"
            ) from exc

    def fetch_trending(self) -> list[dict]:
        """Fetch trending coins on CoinGecko.

        Returns:
            List of dicts with keys: id, name, symbol, market_cap_rank, score.

        Raises:
            DataError: If no trending data.
            NetworkError: If CoinGecko unreachable or circuit breaker open.
        """
        if coingecko_breaker.is_open:
            raise NetworkError(
                "CoinGecko circuit breaker is open. Try again later."
            )

        url = f"{BASE_URL}/search/trending"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            coins = data.get("coins", [])
            if not coins:
                coingecko_breaker.record_success()
                raise DataError("No trending coins from CoinGecko")

            result = []
            for entry in coins:
                item = entry.get("item", {})
                result.append(
                    {
                        "id": item.get("id", ""),
                        "name": item.get("name", ""),
                        "symbol": item.get("symbol", ""),
                        "market_cap_rank": item.get("market_cap_rank", 0),
                        "score": item.get("score", 0),
                    }
                )

            coingecko_breaker.record_success()
            logger.info("CoinGecko trending: %d coins", len(result))
            return result

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            coingecko_breaker.record_failure()
            raise DataError(
                f"CoinGecko trending fetch failed: {exc}"
            ) from exc

    def fetch_categories(self) -> list[dict]:
        """Fetch cryptocurrency categories with market data.

        Returns:
            List of dicts with keys: id, name, market_cap, volume_24h,
            market_cap_change_24h.

        Raises:
            DataError: If no categories data.
            NetworkError: If CoinGecko unreachable or circuit breaker open.
        """
        if coingecko_breaker.is_open:
            raise NetworkError(
                "CoinGecko circuit breaker is open. Try again later."
            )

        url = f"{BASE_URL}/coins/categories"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            if not data:
                coingecko_breaker.record_success()
                raise DataError("No categories from CoinGecko")

            result = []
            for cat in data:
                result.append(
                    {
                        "id": cat.get("id", ""),
                        "name": cat.get("name", ""),
                        "market_cap": float(cat.get("market_cap", 0) or 0),
                        "volume_24h": float(
                            cat.get("total_volume", 0) or 0
                        ),
                        "market_cap_change_24h": float(
                            cat.get("market_cap_change_24h", 0) or 0
                        ),
                    }
                )

            coingecko_breaker.record_success()
            logger.info("CoinGecko categories: %d", len(result))
            return result

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            coingecko_breaker.record_failure()
            raise DataError(
                f"CoinGecko categories fetch failed: {exc}"
            ) from exc
