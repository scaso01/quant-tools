"""Fear & Greed index clients for market sentiment.

- Crypto Fear & Greed Index: REST API from alternative.me (free, no auth)
- CNN Fear & Greed Index: via the ``fear-and-greed`` package
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

logger = logging.getLogger(__name__)

_breaker = CircuitBreaker("fear_greed", threshold=3, recovery_seconds=60.0)

CRYPTO_FNG_URL = "https://api.alternative.me/fng/"


class FearGreedClient:
    """Fetch Fear & Greed indices for crypto and traditional markets.

    Crypto index requires no dependencies (pure REST).
    CNN index requires: ``pip install fear-and-greed``
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._cnn_client = None  # lazy-loaded fear_and_greed module

    def _ensure_cnn_client(self) -> None:
        """Lazy-import the fear-and-greed package."""
        if self._cnn_client is not None:
            return
        try:
            import fear_and_greed

            self._cnn_client = fear_and_greed
        except ImportError:
            raise DataError(
                "fear-and-greed not installed. pip install fear-and-greed"
            )

    def fetch_crypto_index(self, limit: int = 30) -> pd.DataFrame:
        """Fetch the Crypto Fear & Greed Index history.

        Args:
            limit: Number of days to fetch (max ~365 typically available).

        Returns:
            DataFrame with columns: date, value (0-100), classification.
            Classification values: Extreme Fear, Fear, Neutral, Greed,
            Extreme Greed.

        Raises:
            DataError: If no data returned.
            NetworkError: If API unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "Fear & Greed circuit breaker is open. Try again later."
            )

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(
                    CRYPTO_FNG_URL,
                    params={"limit": str(limit), "format": "json"},
                )
                resp.raise_for_status()
                data = resp.json()

            if "data" not in data or not data["data"]:
                _breaker.record_success()
                raise DataError("No crypto Fear & Greed data returned")

            rows = []
            for entry in data["data"]:
                rows.append(
                    {
                        "date": pd.to_datetime(
                            int(entry["timestamp"]), unit="s"
                        ).strftime("%Y-%m-%d"),
                        "value": int(entry["value"]),
                        "classification": entry["value_classification"],
                    }
                )

            _breaker.record_success()
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            logger.info(
                "Crypto Fear & Greed: %d days fetched (latest: %s = %d)",
                len(df),
                df["date"].iloc[-1],
                df["value"].iloc[-1],
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"Crypto Fear & Greed API timeout: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            _breaker.record_failure()
            raise NetworkError(
                f"Crypto Fear & Greed API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            _breaker.record_failure()
            raise DataError(
                f"Crypto Fear & Greed fetch failed: {exc}"
            ) from exc

    def fetch_cnn_index(self) -> dict:
        """Fetch the CNN Fear & Greed Index (current snapshot).

        Returns:
            Dict with keys: score (0-100), rating (str), last_update (str).

        Raises:
            DataError: If library missing or fetch fails.
            NetworkError: If CNN is unreachable or circuit breaker open.
        """
        if _breaker.is_open:
            raise NetworkError(
                "Fear & Greed circuit breaker is open. Try again later."
            )

        self._ensure_cnn_client()

        try:
            result = self._cnn_client.get()

            _breaker.record_success()
            # fear-and-greed returns a namedtuple of exactly
            # (value, description, last_update). Reading previous_close off it
            # raised AttributeError, which surfaced as a fetch failure.
            output = {
                "score": float(result.value),
                "rating": str(result.description),
                "last_update": str(result.last_update),
            }
            logger.info(
                "CNN Fear & Greed: score=%.1f rating=%s",
                output["score"],
                output["rating"],
            )
            return output

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            _breaker.record_failure()
            msg = str(exc).lower()
            if any(
                kw in msg
                for kw in ("timeout", "connection", "refused", "network")
            ):
                raise NetworkError(
                    f"CNN Fear & Greed unreachable: {exc}"
                ) from exc
            raise DataError(
                f"CNN Fear & Greed fetch failed: {exc}"
            ) from exc
