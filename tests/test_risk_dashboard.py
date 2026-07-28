"""Tests for the Risk Dashboard view and risk helpers."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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
    mock_st.slider.return_value = 0.95
    mock_st.selectbox.return_value = "Historical"
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col_mock, col_mock, col_mock, col_mock]

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st


# =============================================================================
# Tests
# =============================================================================


class TestRiskDashboard:
    """Test Risk Dashboard render() and helper functions."""

    def test_render_does_not_crash(self, _mock_streamlit):
        """render() should execute without raising."""
        from quant_tools.dashboard.views.risk_dashboard import render
        render()

    def test_render_calls_header(self, _mock_streamlit):
        """render() should set page header."""
        from quant_tools.dashboard.views.risk_dashboard import render
        render()
        _mock_streamlit.header.assert_called_with("Risk Dashboard")

    def test_compute_var_historical(self):
        """Historical VaR should return a positive number."""
        from quant_tools.dashboard.views.risk_dashboard import compute_var

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        var = compute_var(returns, confidence=0.95, method="historical")
        assert var > 0

    def test_compute_cvar(self):
        """CVaR should be >= VaR."""
        from quant_tools.dashboard.views.risk_dashboard import compute_cvar, compute_var

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        var = compute_var(returns, confidence=0.95)
        cvar = compute_cvar(returns, confidence=0.95)
        assert cvar >= var - 1e-10  # Small tolerance for float

    def test_compute_concentration(self):
        """HHI should be 1.0 for single-asset portfolio."""
        from quant_tools.dashboard.views.risk_dashboard import compute_concentration

        single = {"AAPL": 1.0}
        assert compute_concentration(single) == pytest.approx(1.0)

        # Equal weight across 4 assets -> HHI = 4 * (0.25)^2 = 0.25
        equal = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        assert compute_concentration(equal) == pytest.approx(0.25)

    def test_compute_var_empty(self):
        """VaR of empty series should be 0."""
        from quant_tools.dashboard.views.risk_dashboard import compute_var

        empty = pd.Series(dtype=float)
        assert compute_var(empty) == 0.0

    def test_concentration_empty(self):
        """Concentration of empty weights should be 0."""
        from quant_tools.dashboard.views.risk_dashboard import compute_concentration
        assert compute_concentration({}) == 0.0
