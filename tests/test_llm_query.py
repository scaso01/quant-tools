"""Tests for PortfolioQueryEngine.

All LLM calls are mocked. No real network or model loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.errors import NetworkError
from quant_tools.llm.query import PortfolioQueryEngine, _breaker


SAMPLE_PORTFOLIO = {"AAPL": 0.30, "MSFT": 0.25, "BND": 0.25, "GLD": 0.20}


class TestPortfolioQueryEngine:
    """Tests for PortfolioQueryEngine."""

    def setup_method(self) -> None:
        _breaker.reset()

    # ----- LLM path (mocked) -----

    @patch("httpx.Client")
    def test_query_with_llm(self, mock_client_cls: MagicMock) -> None:
        """Mock httpx POST, verify response extraction."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Tech exposure is 55%."}}],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        engine = PortfolioQueryEngine(base_url="http://fake:8080")
        result = engine.query("What is my tech exposure?", SAMPLE_PORTFOLIO)
        assert result == "Tech exposure is 55%."
        mock_client.post.assert_called_once()

    @patch("httpx.Client")
    def test_llm_error_falls_back(self, mock_client_cls: MagicMock) -> None:
        """Mock httpx to raise, verify fallback used."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("refused")
        mock_client_cls.return_value = mock_client

        engine = PortfolioQueryEngine(base_url="http://fake:8080")
        result = engine.query("What is my exposure?", SAMPLE_PORTFOLIO)
        # Should hit keyword fallback
        assert "AAPL" in result
        assert "Portfolio weights" in result

    # ----- Fallback path -----

    def test_query_fallback_exposure(self) -> None:
        """Test keyword fallback for exposure question."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query(
            "What is my exposure?", SAMPLE_PORTFOLIO, None,
        )
        assert "AAPL" in result
        assert "Portfolio weights" in result

    def test_query_fallback_weight(self) -> None:
        """Test keyword fallback for weight question."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query(
            "Show me portfolio weights", SAMPLE_PORTFOLIO, None,
        )
        assert "30.0%" in result
        assert "MSFT" in result

    def test_query_fallback_top(self) -> None:
        """Test keyword fallback for top holdings."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query(
            "What are my top holdings?", SAMPLE_PORTFOLIO, None,
        )
        assert "Top holdings" in result
        assert "AAPL" in result

    def test_query_fallback_diversification(self) -> None:
        """Test diversification fallback."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query(
            "How diversified is my portfolio?", SAMPLE_PORTFOLIO, None,
        )
        assert "4 positions" in result
        assert "30.0%" in result

    def test_query_fallback_unknown(self) -> None:
        """Unknown question without LLM returns help text."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query(
            "What color is the sky?", SAMPLE_PORTFOLIO, None,
        )
        assert "Unable to answer" in result

    # ----- Circuit breaker -----

    def test_circuit_breaker_open(self) -> None:
        """Test behavior when LLM breaker is open."""
        for _ in range(5):
            _breaker.record_failure()

        engine = PortfolioQueryEngine()
        # _llm_query should raise NetworkError
        with pytest.raises(NetworkError, match="circuit breaker"):
            engine._llm_query("test", SAMPLE_PORTFOLIO, None)

    # ----- Prompt building -----

    def test_build_system_prompt(self) -> None:
        """Verify prompt includes portfolio data."""
        engine = PortfolioQueryEngine()
        prompt = engine._build_system_prompt(
            SAMPLE_PORTFOLIO, {"returns": "+5%"},
        )
        assert "AAPL" in prompt
        assert "quantitative finance" in prompt
        assert "returns" in prompt
        assert "+5%" in prompt

    def test_build_system_prompt_no_context(self) -> None:
        """Prompt without extra context still has portfolio."""
        engine = PortfolioQueryEngine()
        prompt = engine._build_system_prompt(SAMPLE_PORTFOLIO, None)
        assert "AAPL" in prompt
        assert "0.3" in prompt

    # ----- Edge cases -----

    def test_empty_portfolio(self) -> None:
        """Edge case: empty portfolio dict."""
        engine = PortfolioQueryEngine()
        result = engine._fallback_query("What is my exposure?", {}, None)
        assert "Portfolio weights" in result

    def test_default_url(self) -> None:
        """Default URL is localhost:8080."""
        engine = PortfolioQueryEngine()
        assert engine._base_url == "http://localhost:8080"

    def test_custom_url(self) -> None:
        """Custom URL is stored."""
        engine = PortfolioQueryEngine(base_url="http://custom:9999")
        assert engine._base_url == "http://custom:9999"
