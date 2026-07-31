"""Options chain provider via yfinance.

Standalone helper (not a full DataProvider) for fetching options data,
computing put/call ratios, max pain, and scanning unusual volume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import options_breaker

log = logging.getLogger("quant_tools")


@dataclass(frozen=True)
class OptionsContract:
    """A single options contract with Greeks."""

    strike: float
    expiration: str
    option_type: str  # "call" or "put"
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


@dataclass
class OptionsChain:
    """Options chain for a symbol at a given expiration."""

    symbol: str
    expiration: str
    calls: list[OptionsContract] = field(default_factory=list)
    puts: list[OptionsContract] = field(default_factory=list)


class OptionsProvider:
    """Fetch and analyze options data via yfinance.

    Uses yfinance.Ticker for chain data. No API key required.
    """

    def fetch_chain(self, symbol: str, expiration: str | None = None) -> OptionsChain:
        """Fetch options chain via yfinance.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            expiration: Specific expiration date string. If None, uses nearest.

        Returns:
            OptionsChain with calls and puts.

        Raises:
            DataError: If no options data available.
            NetworkError: If the circuit breaker is open.
        """
        if options_breaker.is_open:
            raise NetworkError(f"Options circuit breaker is open for {symbol}")

        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError(
                "yfinance is required for options data. Install with: pip install yfinance"
            ) from exc

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                options_breaker.record_failure()
                raise DataError(f"No options expirations found for {symbol}")

            exp = expiration if expiration and expiration in expirations else expirations[0]
            chain = ticker.option_chain(exp)

            calls = self._convert_contracts(chain.calls, exp, "call")
            puts = self._convert_contracts(chain.puts, exp, "put")

            options_breaker.record_success()
            log.info(
                "Options: %s exp %s — %d calls, %d puts",
                symbol, exp, len(calls), len(puts),
            )
            return OptionsChain(symbol=symbol, expiration=exp, calls=calls, puts=puts)

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            options_breaker.record_failure()
            raise DataError(f"Failed to fetch options for {symbol}: {exc}") from exc

    def fetch_expirations(self, symbol: str) -> list[str]:
        """List available expiration dates for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            List of expiration date strings.

        Raises:
            DataError: If no expirations found.
            NetworkError: If the circuit breaker is open.
        """
        if options_breaker.is_open:
            raise NetworkError(f"Options circuit breaker is open for {symbol}")

        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError(
                "yfinance is required for options data. Install with: pip install yfinance"
            ) from exc

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                options_breaker.record_failure()
                raise DataError(f"No options expirations found for {symbol}")

            options_breaker.record_success()
            return list(expirations)

        except (DataError, NetworkError):
            raise
        except Exception as exc:
            options_breaker.record_failure()
            raise DataError(f"Failed to fetch expirations for {symbol}: {exc}") from exc

    def calculate_put_call_ratio(self, chain: OptionsChain) -> float:
        """Calculate put/call volume ratio.

        Args:
            chain: OptionsChain with calls and puts.

        Returns:
            Ratio of put volume to call volume. >1 = bearish, <1 = bullish.

        Raises:
            DataError: If no volume data available.
        """
        call_volume = sum(c.volume for c in chain.calls if c.volume > 0)
        put_volume = sum(p.volume for p in chain.puts if p.volume > 0)

        if call_volume == 0 and put_volume == 0:
            raise DataError(f"No volume data available for {chain.symbol}")

        if call_volume == 0:
            return float("inf")

        return put_volume / call_volume

    def calculate_max_pain(self, chain: OptionsChain) -> float:
        """Calculate the max pain strike price.

        Max pain is the price at which the total dollar value of outstanding
        options would cause the maximum financial loss to option holders.

        Args:
            chain: OptionsChain with calls and puts.

        Returns:
            Strike price where total option holder losses are maximized.

        Raises:
            DataError: If chain is empty.
        """
        all_contracts = chain.calls + chain.puts
        if not all_contracts:
            raise DataError(f"No contracts in chain for {chain.symbol}")

        strikes = sorted({c.strike for c in all_contracts})
        if not strikes:
            raise DataError(f"No strikes found for {chain.symbol}")

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for test_strike in strikes:
            total_pain = 0.0

            # Pain to call holders: max(0, strike - test_price) * OI
            for call in chain.calls:
                if call.open_interest > 0:
                    itm = max(0.0, test_strike - call.strike)
                    total_pain += itm * call.open_interest

            # Pain to put holders: max(0, test_price - strike) * OI  (wrong direction)
            # Actually: put holder loses when price is ABOVE strike
            # Put holder paid premium, loses when price > strike (OTM)
            # Pain = max(0, strike - test_price) for puts
            for put in chain.puts:
                if put.open_interest > 0:
                    itm = max(0.0, put.strike - test_strike)
                    total_pain += itm * put.open_interest

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    def scan_unusual_volume(
        self, chain: OptionsChain, threshold: float = 2.0
    ) -> list[dict]:
        """Find contracts with unusually high volume relative to open interest.

        Args:
            chain: OptionsChain to scan.
            threshold: Minimum volume/OI ratio to flag (default 2.0).

        Returns:
            List of dicts with contract details and volume/OI ratio.
        """
        unusual: list[dict] = []

        for contract in chain.calls + chain.puts:
            if contract.open_interest > 0 and contract.volume > 0:
                ratio = contract.volume / contract.open_interest
                if ratio >= threshold:
                    unusual.append({
                        "strike": contract.strike,
                        "expiration": contract.expiration,
                        "type": contract.option_type,
                        "volume": contract.volume,
                        "open_interest": contract.open_interest,
                        "vol_oi_ratio": round(ratio, 2),
                        "implied_volatility": contract.implied_volatility,
                    })

        return sorted(unusual, key=lambda x: x["vol_oi_ratio"], reverse=True)

    @staticmethod
    def _convert_contracts(
        df: pd.DataFrame, expiration: str, option_type: str
    ) -> list[OptionsContract]:
        """Convert a yfinance options DataFrame to OptionsContract list.

        Args:
            df: DataFrame from yfinance option_chain().calls or .puts.
            expiration: Expiration date string.
            option_type: "call" or "put".

        Returns:
            List of OptionsContract dataclasses.
        """
        contracts: list[OptionsContract] = []
        if df is None or df.empty:
            return contracts

        for _, row in df.iterrows():
            try:
                contracts.append(OptionsContract(
                    strike=float(row.get("strike", 0)),
                    expiration=expiration,
                    option_type=option_type,
                    last_price=float(row.get("lastPrice", 0)),
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    volume=int(row.get("volume", 0) or 0),
                    open_interest=int(row.get("openInterest", 0) or 0),
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                ))
            except (ValueError, TypeError) as exc:
                log.debug("Skipping malformed contract row: %s", exc)
                continue

        return contracts
