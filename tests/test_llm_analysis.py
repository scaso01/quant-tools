"""Tests for MarketAnalyzer.

All LLM calls are mocked. No real network or model loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.errors import NetworkError
from quant_tools.llm.analysis import MarketAnalyzer, _breaker


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Generate a small OHLCV DataFrame (100 trading days)."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    """OHLCV with only 3 rows (too short for 5d/20d returns)."""
    dates = pd.bdate_range(start="2024-01-02", periods=3)
    return pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 2_000_000, 3_000_000],
        },
        index=dates,
    )


class TestMarketAnalyzer:
    """Tests for MarketAnalyzer."""

    def setup_method(self) -> None:
        _breaker.reset()

    # ----- LLM path (mocked) -----

    @patch("httpx.Client")
    def test_analyze_with_llm(
        self, mock_client_cls: MagicMock, ohlcv_df: pd.DataFrame,
    ) -> None:
        """Mock httpx POST, verify LLM response returned."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "AAPL looks bullish."}}],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        analyzer = MarketAnalyzer(base_url="http://fake:8080")
        result = analyzer.analyze("AAPL", ohlcv_df)
        assert result == "AAPL looks bullish."

    @patch("httpx.Client")
    def test_analyze_llm_error_falls_back(
        self, mock_client_cls: MagicMock, ohlcv_df: pd.DataFrame,
    ) -> None:
        """When LLM fails, fallback template is used."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("refused")
        mock_client_cls.return_value = mock_client

        analyzer = MarketAnalyzer(base_url="http://fake:8080")
        result = analyzer.analyze("AAPL", ohlcv_df)
        assert "AAPL Analysis:" in result
        assert "Last Price:" in result

    # ----- Fallback path -----

    def test_fallback_analysis_bullish(self, ohlcv_df: pd.DataFrame) -> None:
        """Template analysis with realistic data."""
        analyzer = MarketAnalyzer()
        result = analyzer._fallback_analysis("AAPL", ohlcv_df, None)
        assert "AAPL Analysis:" in result
        assert "Last Price: $" in result
        assert "5-Day Return:" in result
        assert "20-Day Return:" in result
        assert "Trend:" in result

    def test_fallback_analysis_with_indicators(
        self, ohlcv_df: pd.DataFrame,
    ) -> None:
        """Template includes indicator data when provided."""
        indicators = {"RSI": 65.3, "MACD": "bullish crossover"}
        analyzer = MarketAnalyzer()
        result = analyzer._fallback_analysis("AAPL", ohlcv_df, indicators)
        assert "RSI" in result
        assert "65.3" in result
        assert "MACD" in result

    def test_fallback_analysis_empty_df(self) -> None:
        """Empty DataFrame returns 'No data' message."""
        analyzer = MarketAnalyzer()
        result = analyzer._fallback_analysis("AAPL", pd.DataFrame(), None)
        assert "No data available for AAPL" in result

    def test_fallback_analysis_none_df(self) -> None:
        """None DataFrame returns 'No data' message."""
        analyzer = MarketAnalyzer()
        result = analyzer._fallback_analysis("AAPL", None, None)
        assert "No data available for AAPL" in result

    def test_fallback_analysis_short_data(
        self, short_ohlcv: pd.DataFrame,
    ) -> None:
        """Short data (< 6 rows) uses zero for 5d return."""
        analyzer = MarketAnalyzer()
        result = analyzer._fallback_analysis("SPY", short_ohlcv, None)
        assert "SPY Analysis:" in result
        assert "5-Day Return: +0.0%" in result

    # ----- Daily summary -----

    def test_daily_summary(self, ohlcv_df: pd.DataFrame) -> None:
        """Daily summary includes all symbols."""
        analyzer = MarketAnalyzer()
        result = analyzer.daily_summary(
            ["AAPL", "MSFT"],
            market_data={"AAPL": ohlcv_df, "MSFT": ohlcv_df},
        )
        assert "Daily Market Summary" in result
        assert "AAPL Analysis:" in result
        assert "MSFT Analysis:" in result

    def test_daily_summary_missing_data(self) -> None:
        """Symbols without data get 'No data' message."""
        analyzer = MarketAnalyzer()
        result = analyzer.daily_summary(["AAPL"], market_data={})
        assert "AAPL: No data available" in result

    def test_daily_summary_no_market_data(self) -> None:
        """None market_data still produces output."""
        analyzer = MarketAnalyzer()
        result = analyzer.daily_summary(["AAPL", "MSFT"])
        assert "AAPL: No data available" in result

    # ----- Circuit breaker -----

    def test_circuit_breaker_open(self, ohlcv_df: pd.DataFrame) -> None:
        """When breaker is open, _llm_analyze raises NetworkError."""
        for _ in range(5):
            _breaker.record_failure()

        analyzer = MarketAnalyzer()
        with pytest.raises(NetworkError, match="circuit breaker"):
            analyzer._llm_analyze("AAPL", ohlcv_df, None, None)

    # ----- Prompt building -----

    def test_build_analysis_prompt(self, ohlcv_df: pd.DataFrame) -> None:
        """Prompt includes indicators and news."""
        analyzer = MarketAnalyzer()
        prompt = analyzer._build_analysis_prompt(
            "AAPL",
            ohlcv_df,
            {"RSI": 70},
            ["Apple beats earnings", "New iPhone announced"],
        )
        assert "Analyze AAPL" in prompt
        assert "RSI" in prompt
        assert "Apple beats earnings" in prompt
        assert "Last close: $" in prompt

    def test_build_analysis_prompt_minimal(
        self, ohlcv_df: pd.DataFrame,
    ) -> None:
        """Prompt with no indicators or news."""
        analyzer = MarketAnalyzer()
        prompt = analyzer._build_analysis_prompt("SPY", ohlcv_df, None, None)
        assert "Analyze SPY" in prompt
        assert "Last close: $" in prompt
        assert "Technical indicators" not in prompt
        assert "Recent headlines" not in prompt
