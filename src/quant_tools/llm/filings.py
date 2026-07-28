"""SEC filing analysis using RAG pipeline with local LLM.

Fetches filings via :class:`SecEdgarClient`, chunks text, and queries the
LLM for summaries, diffs, and guidance extraction.
"""

from __future__ import annotations

import logging

from quant_tools.data.providers.sec_edgar import SecEdgarClient
from quant_tools.errors import DataError, NetworkError
from quant_tools.retry import CircuitBreaker

log = logging.getLogger("quant_tools")
_breaker = CircuitBreaker("llm_filings", threshold=3, recovery_seconds=30.0)

DEFAULT_LLM_URL = "http://localhost:8080"


class FilingAnalyzer:
    """SEC filing analysis using RAG pipeline with local LLM.

    Args:
        base_url: LLM server base URL.  Defaults to ``localhost:8080``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or DEFAULT_LLM_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_filing(
        self, ticker: str, filing_type: str = "10-K"
    ) -> str:
        """Analyse the latest filing.  RAG: fetch -> chunk -> query LLM.

        Falls back to the first 1000 chars of the raw filing text when
        the LLM is unavailable.

        Args:
            ticker: Stock ticker symbol.
            filing_type: ``10-K``, ``10-Q``, or ``8-K``.

        Returns:
            Summary string.
        """
        client = SecEdgarClient()
        filings = client.fetch_filings(ticker, filing_type=filing_type, num_filings=1)
        if not filings:
            raise DataError(f"No {filing_type} filings found for {ticker}")

        text = filings[0].content
        chunks = self._chunk_text(text)

        try:
            system = (
                "You are a financial analyst.  Summarize the key points of "
                "this SEC filing in 5-7 bullet points.  Focus on revenue, "
                "earnings, risk factors, and forward guidance."
            )
            # Use first several chunks to stay within context limits
            user = f"Filing type: {filing_type}\nTicker: {ticker}\n\n"
            user += "\n---\n".join(chunks[:5])
            return self._query_llm(system, user)
        except (NetworkError, Exception):
            # Fallback: return truncated raw text
            return self._fallback_summary(ticker, filing_type, text)

    def diff_filings(
        self, ticker: str, filing_type: str = "10-K", count: int = 2
    ) -> str:
        """Compare consecutive filings, highlighting changes.

        Args:
            ticker: Stock ticker symbol.
            filing_type: Filing type (``10-K``, ``10-Q``, ``8-K``).
            count: Number of filings to compare (default 2).

        Returns:
            Diff summary string.
        """
        client = SecEdgarClient()
        filings = client.fetch_filings(
            ticker, filing_type=filing_type, num_filings=count,
        )
        if len(filings) < 2:
            raise DataError(
                f"Need at least 2 {filing_type} filings for {ticker}, "
                f"got {len(filings)}"
            )

        chunks_old = self._chunk_text(filings[-1].content)
        chunks_new = self._chunk_text(filings[0].content)

        try:
            system = (
                "You are a financial analyst.  Compare these two SEC filings "
                "and highlight the most important changes in revenue, risk "
                "factors, and strategy.  Be concise."
            )
            user = (
                f"Ticker: {ticker}, Filing type: {filing_type}\n\n"
                f"=== OLDER FILING ===\n"
                + "\n---\n".join(chunks_old[:3])
                + f"\n\n=== NEWER FILING ===\n"
                + "\n---\n".join(chunks_new[:3])
            )
            return self._query_llm(system, user)
        except (NetworkError, Exception):
            return (
                f"Unable to diff {filing_type} filings for {ticker} "
                f"(LLM unavailable).  Fetched {len(filings)} filings."
            )

    def extract_guidance(self, ticker: str) -> str:
        """Extract forward guidance from recent 8-K filings.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Guidance summary string.
        """
        client = SecEdgarClient()
        filings = client.fetch_filings(
            ticker, filing_type="8-K", num_filings=3,
        )
        if not filings:
            raise DataError(f"No 8-K filings found for {ticker}")

        combined = "\n---\n".join(f.content[:2000] for f in filings)
        chunks = self._chunk_text(combined)

        try:
            system = (
                "You are a financial analyst.  Extract any forward-looking "
                "guidance (revenue forecasts, earnings outlook, strategic "
                "plans) from these 8-K filings.  List each item as a bullet."
            )
            user = f"Ticker: {ticker}\n\n" + "\n---\n".join(chunks[:5])
            return self._query_llm(system, user)
        except (NetworkError, Exception):
            return (
                f"Unable to extract guidance for {ticker} "
                f"(LLM unavailable).  Found {len(filings)} 8-K filings."
            )

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 2000,
        overlap: int = 200,
    ) -> list[str]:
        """Split *text* into overlapping chunks.

        Args:
            text: Raw filing text.
            chunk_size: Target characters per chunk.
            overlap: Character overlap between consecutive chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _query_llm(self, system: str, user: str) -> str:
        """Query LLM with circuit breaker protection."""
        if _breaker.is_open:
            raise NetworkError("Filing LLM circuit breaker open")

        import httpx

        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                _breaker.record_success()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            _breaker.record_failure()
            raise NetworkError(f"Filing LLM error: {exc}") from exc

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_summary(
        self, ticker: str, filing_type: str, text: str
    ) -> str:
        """Return a truncated raw-text summary when LLM is unavailable."""
        preview = text[:1000].strip()
        return (
            f"{ticker} {filing_type} Summary (LLM unavailable):\n\n"
            f"{preview}\n\n[Truncated -- {len(text)} chars total]"
        )
