"""E2E Playwright tests for the Rolling Analytics page.

Covers widget defaults, sidebar interactions, and negative/edge-case
inputs for the rolling-window metrics page.
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

    def test_symbol_picker_defaults_to_aapl(self, st_page) -> None:
        """The page uses the shared symbol picker, starting on AAPL."""
        assert st_page.symbol_picker_count() == 1
        assert "AAPL" in st_page.get_visible_text()

    def test_window_slider_default(self, st_page) -> None:
        """Window slider should default to 63 days."""
        value = st_page.get_slider_value("Window (days)", sidebar=True)
        assert value == 63

    def test_benchmark_default(self, st_page) -> None:
        """Benchmark selectbox should default to SPY."""
        value = st_page.get_selectbox_value("Benchmark", sidebar=True)
        assert value == "SPY"

    def test_rolling_charts_render(self, st_page) -> None:
        """The page should compute real metrics, not show a placeholder.

        How much history the data layer returns is not under the test's control,
        so a short-history run is an acceptable outcome. What is asserted is that
        the page either draws its metrics or explains why it cannot, and that it
        never falls back to the old "In Development" card.
        """
        st_page.click_button("Run Analysis", sidebar=True, timeout=120_000)
        visible = st_page.get_visible_text()
        assert "In Development" not in visible, "Page is still a placeholder"

        computed = st_page.has_plotly_chart() and st_page.get_metric_count() >= 1
        explained = "Not enough history" in visible
        assert computed or explained, (
            "Expected rolling charts and metric cards, or a clear explanation "
            f"of why they could not be computed. Saw: {visible[-200:]!r}"
        )


# -- TestRollingNegative -------------------------------------------------------


class TestRollingNegative:
    """Negative and boundary tests for the Rolling Analytics page."""

    def test_bogus_symbol_offers_no_suggestion(self, st_page) -> None:
        """The picker only offers real symbols, so junk cannot be submitted."""
        assert st_page.get_symbol_suggestions("ZZZZZ99") == []
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
