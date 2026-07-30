"""AI-powered market analysis via local LLM.

Generates narrative analysis from OHLCV data, technical indicators, and
news headlines.  Falls back to template-based summaries when the LLM is
unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_tools.errors import NetworkError
from quant_tools.retry import CircuitBreaker

log = logging.getLogger("quant_tools")
_breaker = CircuitBreaker("llm_analysis", threshold=3, recovery_seconds=30.0)

DEFAULT_LLM_URL = "http://localhost:8080"


class MarketAnalyzer:
    """AI-powered market analysis via local LLM.

    Args:
        base_url: LLM server base URL.  Defaults to ``localhost:8080``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or DEFAULT_LLM_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        indicators: dict | None = None,
        news: list[str] | None = None,
    ) -> str:
        """Generate AI market analysis for *symbol*.

        Args:
            symbol: Ticker symbol.
            ohlcv: OHLCV DataFrame (must contain ``Close``).
            indicators: Optional dict of technical indicators.
            news: Optional list of headline strings.

        Returns:
            Narrative analysis string.
        """
        try:
            return self._llm_analyze(symbol, ohlcv, indicators, news)
        except Exception:
            return self._fallback_analysis(symbol, ohlcv, indicators)

    def daily_summary(
        self,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame] | None = None,
    ) -> str:
        """Generate a daily market summary for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            market_data: Optional ``{symbol: ohlcv_df}`` mapping.

        Returns:
            Multi-symbol summary string.
        """
        parts: list[str] = [f"Daily Market Summary ({len(symbols)} symbols)"]
        parts.append("=" * 50)

        for sym in symbols:
            df = market_data.get(sym) if market_data else None
            if df is not None and len(df) > 0:
                parts.append(self._fallback_analysis(sym, df, None))
            else:
                parts.append(f"{sym}: No data available.")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        indicators: dict | None,
        news: list[str] | None,
    ) -> str:
        """Query the LLM for a narrative analysis."""
        if _breaker.is_open:
            raise NetworkError("Analysis LLM circuit breaker open")

        import httpx

        system = (
            "You are a quantitative market analyst.  Provide a concise, "
            "data-driven analysis (3-5 sentences) based on the data below."
        )
        user = self._build_analysis_prompt(symbol, ohlcv, indicators, news)

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 500,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Some local models end their turn immediately on this system
                # prompt and return a completion with no text. That came back as
                # a successful empty analysis, so the command printed a heading
                # and nothing else. Treat it as a failed call so the caller falls
                # back to the deterministic summary.
                if not content or not content.strip():
                    raise NetworkError("Analysis LLM returned an empty completion")
                _breaker.record_success()
                return content
        except Exception as exc:
            _breaker.record_failure()
            raise NetworkError(f"Analysis LLM error: {exc}") from exc

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_analysis_prompt(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        indicators: dict | None,
        news: list[str] | None,
    ) -> str:
        """Build a detailed analysis prompt with market context."""
        lines: list[str] = [f"Analyze {symbol}:"]

        if ohlcv is not None and len(ohlcv) > 0:
            close = ohlcv["Close"].values
            last = close[-1]
            lines.append(f"  Last close: ${last:.2f}")

            if len(close) > 5:
                ret_5d = (close[-1] / close[-6] - 1) * 100
                lines.append(f"  5-day return: {ret_5d:+.1f}%")
            if len(close) > 20:
                ret_20d = (close[-1] / close[-21] - 1) * 100
                lines.append(f"  20-day return: {ret_20d:+.1f}%")
            if len(close) > 60:
                ret_60d = (close[-1] / close[-61] - 1) * 100
                lines.append(f"  60-day return: {ret_60d:+.1f}%")

            if "Volume" in ohlcv.columns and len(ohlcv) > 5:
                vol_avg = ohlcv["Volume"].iloc[-5:].mean()
                vol_prev = ohlcv["Volume"].iloc[-20:-5].mean() if len(ohlcv) > 20 else vol_avg
                if vol_prev > 0:
                    vol_change = (vol_avg / vol_prev - 1) * 100
                    lines.append(f"  Volume trend (5d vs 20d): {vol_change:+.0f}%")

        if indicators:
            lines.append("  Technical indicators:")
            for key, val in indicators.items():
                lines.append(f"    {key}: {val}")

        if news:
            lines.append("  Recent headlines:")
            for headline in news[:5]:
                lines.append(f"    - {headline}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _fallback_analysis(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        indicators: dict | None,
    ) -> str:
        """Template-based analysis when LLM is unavailable."""
        if ohlcv is None or len(ohlcv) == 0:
            return f"No data available for {symbol}."

        close = ohlcv["Close"].values
        last = close[-1]
        ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
        ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) > 20 else 0

        if ret_5d > 0 and ret_20d > 0:
            trend = "bullish"
        elif ret_5d < 0 and ret_20d < 0:
            trend = "bearish"
        else:
            trend = "mixed"

        parts = [
            f"{symbol} Analysis:",
            f"  Last Price: ${last:.2f}",
            f"  5-Day Return: {ret_5d:+.1f}%",
            f"  20-Day Return: {ret_20d:+.1f}%",
            f"  Trend: {trend}",
        ]

        if indicators:
            for key, val in indicators.items():
                parts.append(f"  {key}: {val}")

        return "\n".join(parts)
