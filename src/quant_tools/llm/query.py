"""Natural language portfolio querying via local LLM.

Routes questions to a local llama-server (OpenAI-compatible) with a
keyword-based fallback when the LLM is unavailable.
"""

from __future__ import annotations

import json
import logging

from quant_tools.errors import NetworkError
from quant_tools.retry import CircuitBreaker

log = logging.getLogger("quant_tools")
_breaker = CircuitBreaker("llm", threshold=3, recovery_seconds=30.0)

# Default LLM endpoint (Beast llama-server)
DEFAULT_LLM_URL = "http://localhost:8080"


class PortfolioQueryEngine:
    """Natural language portfolio querying via local LLM.

    Args:
        base_url: LLM server base URL.  Defaults to ``localhost:8080``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or DEFAULT_LLM_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        portfolio: dict[str, float],
        context: dict | None = None,
    ) -> str:
        """Answer a natural-language question about a portfolio.

        Tries the LLM first; falls back to keyword matching on failure.

        Args:
            question: e.g. "What is my tech exposure?"
            portfolio: ``{symbol: weight}`` dict.
            context: Optional additional data (prices, returns, etc.).

        Returns:
            Formatted answer string.
        """
        try:
            return self._llm_query(question, portfolio, context)
        except Exception:
            return self._fallback_query(question, portfolio, context)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_query(
        self,
        question: str,
        portfolio: dict[str, float],
        context: dict | None,
    ) -> str:
        """Query via llama-server ``/v1/chat/completions``."""
        if _breaker.is_open:
            raise NetworkError("LLM circuit breaker open")

        import httpx

        system_prompt = self._build_system_prompt(portfolio, context)

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        "max_tokens": 500,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                _breaker.record_success()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            _breaker.record_failure()
            raise NetworkError(f"LLM error: {exc}") from exc

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        portfolio: dict[str, float],
        context: dict | None,
    ) -> str:
        """Build a context-rich system prompt from portfolio data."""
        lines = ["You are a quantitative finance assistant. Answer concisely."]
        lines.append(f"\nPortfolio holdings: {json.dumps(portfolio, indent=2)}")
        if context:
            for key, val in context.items():
                lines.append(f"\n{key}: {val}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------

    def _fallback_query(
        self,
        question: str,
        portfolio: dict[str, float],
        context: dict | None,
    ) -> str:
        """Keyword-based fallback when LLM is unavailable."""
        q = question.lower()

        if "exposure" in q or "weight" in q:
            sorted_h = sorted(portfolio.items(), key=lambda x: -x[1])
            lines = [f"  {sym}: {w:.1%}" for sym, w in sorted_h]
            return "Portfolio weights:\n" + "\n".join(lines)

        if "largest" in q or "biggest" in q or "top" in q:
            top = sorted(portfolio.items(), key=lambda x: -x[1])[:5]
            return "Top holdings: " + ", ".join(
                f"{s} ({w:.1%})" for s, w in top
            )

        if "diversif" in q:
            n = len(portfolio)
            max_w = max(portfolio.values()) if portfolio else 0
            return f"Portfolio has {n} positions. Max single position: {max_w:.1%}."

        return (
            "Unable to answer without LLM. "
            "Start llama-server for AI-powered responses."
        )
