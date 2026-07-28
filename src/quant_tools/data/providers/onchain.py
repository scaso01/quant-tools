"""DeFi Llama on-chain data client.

Standalone helper (not a full DataProvider) for fetching DeFi protocol data
from the free DeFi Llama REST API. No API key required.

API docs: https://defillama.com/docs/api
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import defillama_breaker

logger = logging.getLogger("quant_tools")

BASE_URL = "https://api.llama.fi"
YIELDS_URL = "https://yields.llama.fi"
STABLECOINS_URL = "https://stablecoins.llama.fi"


class DefiLlamaClient:
    """Fetch on-chain DeFi data from DeFi Llama.

    No authentication required. Uses httpx for REST calls.

    Args:
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch_tvl(self, protocol: str) -> pd.DataFrame:
        """Fetch TVL (Total Value Locked) history for a protocol.

        Args:
            protocol: Protocol slug (e.g. "aave", "uniswap", "lido").

        Returns:
            DataFrame with columns: date, tvl.

        Raises:
            DataError: If no TVL data found.
            NetworkError: If API is unreachable or circuit breaker is open.
        """
        if defillama_breaker.is_open:
            raise NetworkError("DeFi Llama circuit breaker is open. Try again later.")

        url = f"{BASE_URL}/protocol/{protocol}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            tvl_data = data.get("tvl", [])
            if not tvl_data:
                defillama_breaker.record_failure()
                raise DataError(f"No TVL data found for protocol '{protocol}'")

            rows = []
            for entry in tvl_data:
                rows.append({
                    "date": pd.to_datetime(entry["date"], unit="s"),
                    "tvl": float(entry["totalLiquidityUSD"]),
                })

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            defillama_breaker.record_success()
            logger.info(
                "DeFi Llama: %s TVL — %d data points", protocol, len(df),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            defillama_breaker.record_failure()
            raise NetworkError(f"DeFi Llama API timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            defillama_breaker.record_failure()
            if exc.response.status_code == 400:
                raise DataError(
                    f"Unknown protocol '{protocol}'. Check slug at defillama.com"
                ) from exc
            raise NetworkError(
                f"DeFi Llama API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            defillama_breaker.record_failure()
            raise DataError(f"DeFi Llama fetch failed: {exc}") from exc

    def fetch_yields(
        self, chain: str | None = None, min_tvl: float = 1e6
    ) -> pd.DataFrame:
        """Fetch yield/APY opportunities across DeFi protocols.

        Args:
            chain: Filter by chain (e.g. "Ethereum", "Arbitrum"). None for all.
            min_tvl: Minimum TVL in USD to include (default 1M).

        Returns:
            DataFrame with columns: pool, chain, tvl, apy, project.

        Raises:
            DataError: If no yield data found.
            NetworkError: If API is unreachable or circuit breaker is open.
        """
        if defillama_breaker.is_open:
            raise NetworkError("DeFi Llama circuit breaker is open. Try again later.")

        url = f"{YIELDS_URL}/pools"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            pools = data.get("data", [])
            if not pools:
                defillama_breaker.record_failure()
                raise DataError("No yield pool data returned from DeFi Llama")

            rows = []
            for pool in pools:
                tvl = float(pool.get("tvlUsd", 0) or 0)
                if tvl < min_tvl:
                    continue

                pool_chain = pool.get("chain", "")
                if chain and pool_chain.lower() != chain.lower():
                    continue

                rows.append({
                    "pool": pool.get("symbol", ""),
                    "chain": pool_chain,
                    "tvl": tvl,
                    "apy": float(pool.get("apy", 0) or 0),
                    "project": pool.get("project", ""),
                })

            if not rows:
                defillama_breaker.record_failure()
                filter_msg = f" for chain={chain}" if chain else ""
                raise DataError(
                    f"No yield pools found{filter_msg} with TVL >= ${min_tvl:,.0f}"
                )

            df = pd.DataFrame(rows).sort_values("apy", ascending=False).reset_index(drop=True)
            defillama_breaker.record_success()
            logger.info("DeFi Llama: %d yield pools returned", len(df))
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            defillama_breaker.record_failure()
            raise NetworkError(f"DeFi Llama yields API timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            defillama_breaker.record_failure()
            raise NetworkError(
                f"DeFi Llama yields API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            defillama_breaker.record_failure()
            raise DataError(f"DeFi Llama yields fetch failed: {exc}") from exc

    def fetch_stablecoins(self) -> pd.DataFrame:
        """Fetch stablecoin market data.

        Returns:
            DataFrame with columns: name, symbol, peg, mcap, chain.

        Raises:
            DataError: If no stablecoin data found.
            NetworkError: If API is unreachable or circuit breaker is open.
        """
        if defillama_breaker.is_open:
            raise NetworkError("DeFi Llama circuit breaker is open. Try again later.")

        url = f"{STABLECOINS_URL}/stablecoins"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params={"includePrices": "true"})
                resp.raise_for_status()
                data = resp.json()

            coins = data.get("peggedAssets", [])
            if not coins:
                defillama_breaker.record_failure()
                raise DataError("No stablecoin data returned from DeFi Llama")

            rows = []
            for coin in coins:
                # Get total circulating across all chains
                circulating = coin.get("circulating", {})
                mcap = float(circulating.get("peggedUSD", 0) or 0)

                chains = list(coin.get("chainCirculating", {}).keys())
                chain_str = ", ".join(chains[:3])
                if len(chains) > 3:
                    chain_str += f" +{len(chains) - 3}"

                rows.append({
                    "name": coin.get("name", ""),
                    "symbol": coin.get("symbol", ""),
                    "peg": coin.get("pegType", ""),
                    "mcap": mcap,
                    "chain": chain_str,
                })

            if not rows:
                defillama_breaker.record_failure()
                raise DataError("No stablecoin entries parsed from DeFi Llama")

            df = pd.DataFrame(rows).sort_values("mcap", ascending=False).reset_index(drop=True)
            defillama_breaker.record_success()
            logger.info("DeFi Llama: %d stablecoins returned", len(df))
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            defillama_breaker.record_failure()
            raise NetworkError(f"DeFi Llama stablecoins API timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            defillama_breaker.record_failure()
            raise NetworkError(
                f"DeFi Llama stablecoins API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            defillama_breaker.record_failure()
            raise DataError(f"DeFi Llama stablecoins fetch failed: {exc}") from exc

    def fetch_fees(self, protocol: str) -> pd.DataFrame:
        """Fetch protocol fee and revenue data.

        Args:
            protocol: Protocol slug (e.g. "uniswap", "aave").

        Returns:
            DataFrame with columns: date, fees, revenue.

        Raises:
            DataError: If no fee data found.
            NetworkError: If API is unreachable or circuit breaker is open.
        """
        if defillama_breaker.is_open:
            raise NetworkError("DeFi Llama circuit breaker is open. Try again later.")

        url = f"{BASE_URL}/summary/fees/{protocol}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            total_data = data.get("totalDataChart", [])
            if not total_data:
                defillama_breaker.record_failure()
                raise DataError(f"No fee data found for protocol '{protocol}'")

            # Revenue chart may be separate
            revenue_data = data.get("totalDataChartBreakdown", [])
            revenue_map: dict[int, float] = {}
            for entry in revenue_data:
                if isinstance(entry, list) and len(entry) >= 2:
                    ts = entry[0]
                    breakdown = entry[1] if isinstance(entry[1], dict) else {}
                    rev = sum(
                        v.get("Revenue", 0) or 0
                        for v in breakdown.values()
                        if isinstance(v, dict)
                    )
                    revenue_map[ts] = rev

            rows = []
            for entry in total_data:
                if isinstance(entry, list) and len(entry) >= 2:
                    ts = entry[0]
                    fees = float(entry[1])
                    rows.append({
                        "date": pd.to_datetime(ts, unit="s"),
                        "fees": fees,
                        "revenue": revenue_map.get(ts, 0.0),
                    })

            if not rows:
                defillama_breaker.record_failure()
                raise DataError(f"No fee entries parsed for protocol '{protocol}'")

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            defillama_breaker.record_success()
            logger.info(
                "DeFi Llama: %s fees — %d data points", protocol, len(df),
            )
            return df

        except (DataError, NetworkError):
            raise
        except httpx.TimeoutException as exc:
            defillama_breaker.record_failure()
            raise NetworkError(f"DeFi Llama fees API timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            defillama_breaker.record_failure()
            if exc.response.status_code == 400:
                raise DataError(
                    f"Unknown protocol '{protocol}' for fees. Check slug at defillama.com"
                ) from exc
            raise NetworkError(
                f"DeFi Llama fees API error {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            defillama_breaker.record_failure()
            raise DataError(f"DeFi Llama fees fetch failed: {exc}") from exc
