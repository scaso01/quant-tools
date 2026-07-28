"""Tests for the Rolling Analytics dashboard view."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Mock streamlit before importing the module
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_streamlit():
    """Mock streamlit module for all tests in this file."""
    mock_st = MagicMock()
    mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
    mock_st.sidebar.__exit__ = MagicMock(return_value=False)
    # text_input and slider return sensible defaults
    mock_st.text_input.return_value = "AAPL"
    mock_st.slider.return_value = 63
    # columns returns list of mocks with context manager support
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col_mock, col_mock]

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st


# =============================================================================
# Tests
# =============================================================================


class TestRollingView:
    """Test Rolling Analytics render() function."""

    def test_render_does_not_crash(self, _mock_streamlit):
        """render() should execute without raising."""
        from quant_tools.dashboard.views.rolling import render
        render()

    def test_render_calls_header(self, _mock_streamlit):
        """render() should call st.header."""
        from quant_tools.dashboard.views.rolling import render
        render()
        _mock_streamlit.header.assert_called()

    def test_render_shows_coming_soon(self, _mock_streamlit):
        """render() should show coming-soon state with markdown content."""
        from quant_tools.dashboard.views.rolling import render
        render()
        _mock_streamlit.markdown.assert_called()

    def test_rolling_sharpe_computation(self):
        """_compute_rolling_sharpe should return valid Series."""
        from quant_tools.dashboard.views.rolling import _compute_rolling_sharpe

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        result = _compute_rolling_sharpe(returns, window=63)
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        assert result.notna().all()

    def test_rolling_volatility_computation(self):
        """_compute_rolling_volatility should return positive values."""
        from quant_tools.dashboard.views.rolling import _compute_rolling_volatility

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        result = _compute_rolling_volatility(returns, window=63)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()
