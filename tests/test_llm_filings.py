"""Tests for FilingAnalyzer.

All SEC EDGAR and LLM calls are mocked. No real network or model loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.data.providers.sec_edgar import Filing
from quant_tools.errors import DataError, NetworkError
from quant_tools.llm.filings import FilingAnalyzer, _breaker


def _make_filing(
    ticker: str = "AAPL",
    filing_type: str = "10-K",
    content: str = "Revenue increased 15%. Risk factors include supply chain.",
) -> Filing:
    """Helper: create a Filing dataclass."""
    return Filing(
        ticker=ticker,
        filing_type=filing_type,
        filename="full-submission.txt",
        content=content,
    )


class TestFilingAnalyzer:
    """Tests for FilingAnalyzer."""

    def setup_method(self) -> None:
        _breaker.reset()

    # ----- chunk_text -----

    def test_chunk_text_basic(self) -> None:
        """Basic chunking with overlap."""
        analyzer = FilingAnalyzer()
        text = "A" * 5000
        chunks = analyzer._chunk_text(text, chunk_size=2000, overlap=200)
        assert len(chunks) >= 3
        # First chunk is 2000 chars
        assert len(chunks[0]) == 2000

    def test_chunk_text_short(self) -> None:
        """Text shorter than chunk_size produces single chunk."""
        analyzer = FilingAnalyzer()
        chunks = analyzer._chunk_text("Short text.", chunk_size=2000)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_chunk_text_empty(self) -> None:
        """Empty text returns empty list."""
        analyzer = FilingAnalyzer()
        assert analyzer._chunk_text("") == []

    def test_chunk_text_overlap(self) -> None:
        """Overlapping chunks share characters at boundaries."""
        analyzer = FilingAnalyzer()
        # Use letters so we can verify overlap
        text = "".join(chr(65 + (i % 26)) for i in range(500))
        chunks = analyzer._chunk_text(text, chunk_size=200, overlap=50)
        # Second chunk should start at offset 150 (200 - 50)
        assert chunks[1][:50] == chunks[0][150:200]

    def test_chunk_text_exact_size(self) -> None:
        """Text exactly chunk_size produces one full chunk + overlap tail."""
        analyzer = FilingAnalyzer()
        text = "X" * 2000
        chunks = analyzer._chunk_text(text, chunk_size=2000, overlap=200)
        # First chunk is 2000 chars, overlap generates a small tail chunk
        assert len(chunks[0]) == 2000
        assert all(len(c) <= 2000 for c in chunks)

    # ----- analyze_filing -----

    @patch("httpx.Client")
    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_analyze_filing_with_llm(
        self, mock_sec_cls: MagicMock, mock_http_cls: MagicMock,
    ) -> None:
        """analyze_filing with mocked SEC + LLM returns LLM response."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = [_make_filing()]
        mock_sec_cls.return_value = mock_sec

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Revenue grew 15%."}}],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_http_cls.return_value = mock_client

        analyzer = FilingAnalyzer(base_url="http://fake:8080")
        result = analyzer.analyze_filing("AAPL", "10-K")
        assert result == "Revenue grew 15%."

    @patch("httpx.Client")
    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_analyze_filing_llm_unavailable_fallback(
        self, mock_sec_cls: MagicMock, mock_http_cls: MagicMock,
    ) -> None:
        """When LLM fails, fallback returns truncated text."""
        content = "Revenue increased." + "X" * 2000
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = [_make_filing(content=content)]
        mock_sec_cls.return_value = mock_sec

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("refused")
        mock_http_cls.return_value = mock_client

        analyzer = FilingAnalyzer(base_url="http://fake:8080")
        result = analyzer.analyze_filing("AAPL", "10-K")
        assert "LLM unavailable" in result
        assert "Revenue increased" in result

    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_analyze_filing_no_filings(
        self, mock_sec_cls: MagicMock,
    ) -> None:
        """No filings found raises DataError."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = []
        mock_sec_cls.return_value = mock_sec

        analyzer = FilingAnalyzer()
        with pytest.raises(DataError, match="No 10-K filings"):
            analyzer.analyze_filing("ZZZZ")

    # ----- diff_filings -----

    @patch("httpx.Client")
    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_diff_filings_success(
        self, mock_sec_cls: MagicMock, mock_http_cls: MagicMock,
    ) -> None:
        """diff_filings with 2 filings and working LLM."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = [
            _make_filing(content="New revenue: $100B"),
            _make_filing(content="Old revenue: $80B"),
        ]
        mock_sec_cls.return_value = mock_sec

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Revenue grew from $80B to $100B."}}],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_http_cls.return_value = mock_client

        analyzer = FilingAnalyzer(base_url="http://fake:8080")
        result = analyzer.diff_filings("AAPL")
        assert "Revenue grew" in result

    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_diff_filings_insufficient(
        self, mock_sec_cls: MagicMock,
    ) -> None:
        """diff_filings with < 2 filings raises DataError."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = [_make_filing()]
        mock_sec_cls.return_value = mock_sec

        analyzer = FilingAnalyzer()
        with pytest.raises(DataError, match="Need at least 2"):
            analyzer.diff_filings("AAPL")

    # ----- extract_guidance -----

    @patch("httpx.Client")
    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_extract_guidance_success(
        self, mock_sec_cls: MagicMock, mock_http_cls: MagicMock,
    ) -> None:
        """extract_guidance with mocked 8-K filings."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = [
            _make_filing(filing_type="8-K", content="Guidance: Q2 revenue $50B"),
        ]
        mock_sec_cls.return_value = mock_sec

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Q2 guidance: $50B revenue."}}],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_http_cls.return_value = mock_client

        analyzer = FilingAnalyzer(base_url="http://fake:8080")
        result = analyzer.extract_guidance("AAPL")
        assert "Q2" in result

    @patch("quant_tools.llm.filings.SecEdgarClient")
    def test_extract_guidance_no_filings(
        self, mock_sec_cls: MagicMock,
    ) -> None:
        """extract_guidance with no 8-K filings raises DataError."""
        mock_sec = MagicMock()
        mock_sec.fetch_filings.return_value = []
        mock_sec_cls.return_value = mock_sec

        analyzer = FilingAnalyzer()
        with pytest.raises(DataError, match="No 8-K filings"):
            analyzer.extract_guidance("ZZZZ")

    # ----- Circuit breaker -----

    def test_circuit_breaker_open(self) -> None:
        """When breaker is open, _query_llm raises NetworkError."""
        for _ in range(5):
            _breaker.record_failure()

        analyzer = FilingAnalyzer()
        with pytest.raises(NetworkError, match="circuit breaker"):
            analyzer._query_llm("system", "user")

    # ----- Fallback summary -----

    def test_fallback_summary(self) -> None:
        """Fallback summary truncates and labels correctly."""
        analyzer = FilingAnalyzer()
        text = "A" * 3000
        result = analyzer._fallback_summary("AAPL", "10-K", text)
        assert "LLM unavailable" in result
        assert "3000 chars total" in result
        # Preview should be 1000 chars
        assert "A" * 100 in result
