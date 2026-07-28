"""E2E Playwright tests for the Rolling Analytics page.

Covers widget defaults, sidebar interactions, and negative/edge-case
inputs for the placeholder rolling-window metrics page.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
ROLLING_PAGE = "\U0001f4c9 Rolling Analytics"


@pytest.fixture(autouse=True)
def _navigate_to_rolling(st_page) -> None:
    """Navigate to the Rolling Analytics page before every test."""
    st_page.navigate_to(ROLLING_PAGE)


# -- TestRollingWidgets --------------------------------------------------------


class TestRollingWidgets:
    """Verify sidebar widgets render with correct default values."""

    def test_page_loads(self, st_page) -> None:
        """Rolling Analytics page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Rolling Analytics" in visible

    def test_ticker_input_default(self, st_page) -> None:
        """Ticker text input should default to AAPL."""
        value = st_page.get_text_input_value("Ticker", sidebar=True)
        assert value == "AAPL"

    def test_window_slider_default(self, st_page) -> None:
        """Window slider should default to 63 days."""
        value = st_page.get_slider_value("Window (days)", sidebar=True)
        assert value == 63

    def test_benchmark_input_default(self, st_page) -> None:
        """Benchmark text input should default to SPY."""
        value = st_page.get_text_input_value("Benchmark", sidebar=True)
        assert value == "SPY"


# -- TestRollingNegative -------------------------------------------------------


class TestRollingNegative:
    """Negative and boundary tests for the Rolling Analytics page."""

    def test_empty_ticker(self, st_page) -> None:
        """Clearing the ticker should not crash the page."""
        st_page.set_text_input("Ticker", "", sidebar=True)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_window_min_boundary(self, st_page) -> None:
        """Setting window slider toward minimum (10) should not crash."""
        original = st_page.get_slider_value("Window (days)", sidebar=True)
        st_page.set_slider("Window (days)", 10, sidebar=True)
        value = st_page.get_slider_value("Window (days)", sidebar=True)
        # Verify the slider moved toward the minimum and no crash
        assert value <= original
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible

    def test_window_max_boundary(self, st_page) -> None:
        """Setting window slider toward maximum (252) should not crash."""
        original = st_page.get_slider_value("Window (days)", sidebar=True)
        st_page.set_slider("Window (days)", 252, sidebar=True)
        value = st_page.get_slider_value("Window (days)", sidebar=True)
        # Verify the slider moved toward the maximum and no crash
        assert value >= original
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible
