"""Tests for the AI Analysis dashboard view."""
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
    mock_st.selectbox.return_value = "Market Overview"
    mock_st.slider.side_effect = [0.7, 1024]  # temperature, max_tokens
    mock_st.text_area.return_value = ""
    mock_st.button.return_value = False  # Don't trigger generation by default
    # columns with context manager support
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col_mock, col_mock]

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st


# =============================================================================
# Tests
# =============================================================================


class TestAIAnalysisView:
    """Test AI Analysis render() function."""

    def test_render_does_not_crash(self, _mock_streamlit):
        """render() should execute without raising."""
        from quant_tools.dashboard.views.ai_analysis import render
        render()

    def test_render_calls_header(self, _mock_streamlit):
        """render() should set page header."""
        from quant_tools.dashboard.views.ai_analysis import render
        render()
        _mock_streamlit.header.assert_called_with("AI Analysis")

    def test_analysis_types_defined(self):
        """ANALYSIS_TYPES should contain expected entries."""
        from quant_tools.dashboard.views.ai_analysis import ANALYSIS_TYPES
        assert "Market Overview" in ANALYSIS_TYPES
        assert "Trade Ideas" in ANALYSIS_TYPES
        assert "Portfolio Review" in ANALYSIS_TYPES
        assert len(ANALYSIS_TYPES) >= 5

    def test_render_without_generation(self, _mock_streamlit):
        """When button not clicked, should show info message."""
        _mock_streamlit.button.return_value = False
        from quant_tools.dashboard.views.ai_analysis import render
        render()
        _mock_streamlit.info.assert_called()

    def test_call_llm_handles_failure(self):
        """_call_llm should return error string when requests fails."""
        from quant_tools.dashboard.views.ai_analysis import _call_llm

        # requests.post is already blocked by conftest network guard
        # so it will raise RuntimeError
        result = _call_llm("test prompt")
        assert "LLM unavailable" in result or "blocked" in result.lower()
