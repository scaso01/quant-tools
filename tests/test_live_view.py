"""Tests for the Live Monitor dashboard view."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock streamlit
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_streamlit():
    """Mock streamlit module for all tests in this file."""
    mock_st = MagicMock()
    mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
    mock_st.sidebar.__exit__ = MagicMock(return_value=False)
    mock_st.text_area.return_value = "AAPL, MSFT, GOOGL"
    mock_st.slider.return_value = 5
    # columns with context manager support
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col_mock, col_mock, col_mock, col_mock, col_mock]

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st


# =============================================================================
# Tests
# =============================================================================


class TestLiveView:
    """Test Live Monitor render() and watchlist parsing."""

    def test_render_does_not_crash(self, _mock_streamlit):
        """render() should execute without raising."""
        from quant_tools.dashboard.views.live import render
        render()

    def test_render_calls_header(self, _mock_streamlit):
        """render() should set page header."""
        from quant_tools.dashboard.views.live import render
        render()
        _mock_streamlit.header.assert_called_with("Live Monitor")

    def test_parse_watchlist_basic(self):
        """parse_watchlist should split comma-separated symbols."""
        from quant_tools.dashboard.views.live import parse_watchlist

        result = parse_watchlist("AAPL, MSFT, GOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_watchlist_deduplication(self):
        """parse_watchlist should remove duplicates."""
        from quant_tools.dashboard.views.live import parse_watchlist

        result = parse_watchlist("AAPL, MSFT, AAPL, msft")
        assert result == ["AAPL", "MSFT"]

    def test_parse_watchlist_empty(self):
        """parse_watchlist should return empty list for empty input."""
        from quant_tools.dashboard.views.live import parse_watchlist

        assert parse_watchlist("") == []
        assert parse_watchlist("  ") == []

    def test_parse_watchlist_newlines(self):
        """parse_watchlist should handle newline-separated symbols."""
        from quant_tools.dashboard.views.live import parse_watchlist

        result = parse_watchlist("AAPL\nMSFT\nGOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_render_shows_coming_soon(self, _mock_streamlit):
        """render() should show coming-soon state with info message."""
        from quant_tools.dashboard.views.live import render
        render()
        _mock_streamlit.info.assert_called()
