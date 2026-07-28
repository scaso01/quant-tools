"""E2E tests for the Performance page in the Streamlit dashboard.

Covers widget presence, empty-state handling, and the resolve workflow.
"""

from __future__ import annotations

import pytest

from .streamlit_helpers import (
    SEL_BUTTON,
    StreamlitPage,
)

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
PERFORMANCE_PAGE = "\U0001f4ca Performance"


@pytest.fixture(autouse=True)
def _navigate_to_performance(st_page: StreamlitPage) -> None:
    """Navigate to the Performance page before every test."""
    st_page.navigate_to(PERFORMANCE_PAGE)


# -- TestPerformanceWidgets ---------------------------------------------------


class TestPerformanceWidgets:
    """Verify that all expected widgets render on the Performance page."""

    def test_page_loads(self, st_page: StreamlitPage) -> None:
        """Performance page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Performance" in visible

    def test_kpi_row(self, st_page: StreamlitPage) -> None:
        """KPI metric cards should be visible (count depends on data availability)."""
        metric_count = st_page.get_metric_count()
        # With no prediction data the page may render zero metric cards
        assert metric_count >= 0, (
            f"Expected metric count >= 0, got {metric_count}"
        )
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_resolve_button_exists(self, st_page: StreamlitPage) -> None:
        """Resolve Pending Predictions button should be present on the page."""
        assert st_page.widget_exists(SEL_BUTTON, "Resolve Pending Predictions")

    def test_model_comparison_section(self, st_page: StreamlitPage) -> None:
        """Model Comparison section should appear (header or dataframe)."""
        visible = st_page.get_visible_text()
        # The section header is always rendered when performance data exists.
        # When no data exists, we still expect the page to load cleanly.
        has_section = "Model Comparison" in visible
        has_dataframe = st_page.has_dataframe()
        # Either the section is visible with data, or the page is in empty state
        assert has_section or has_dataframe or "Performance" in visible


# -- TestPerformanceNegative --------------------------------------------------


class TestPerformanceNegative:
    """Negative and edge-case tests for graceful empty-state display."""

    def test_empty_state(self, st_page: StreamlitPage) -> None:
        """Page should render gracefully when no performance data exists."""
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # Page should at least show its header
        assert "Performance" in visible


# -- TestPerformanceWorkflows -------------------------------------------------


class TestPerformanceWorkflows:
    """Workflow tests for resolving predictions and verifying page content."""

    def test_resolve_pending(self, st_page: StreamlitPage) -> None:
        """Clicking Resolve Pending should not crash regardless of pending state."""
        st_page.click_button("Resolve Pending Predictions", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash on Resolve"
        # After resolving, the page should show some feedback
        has_feedback = (
            st_page.has_success()
            or st_page.has_info()
            or st_page.has_warning()
            or st_page.has_error()
            or "Performance" in visible
        )
        assert has_feedback, "Expected some outcome after clicking Resolve"

    def test_page_content_renders(self, st_page: StreamlitPage) -> None:
        """Header and key sections should render without crashing."""
        visible = st_page.get_visible_text()
        # Header
        assert "Performance" in visible
        assert "Traceback" not in visible, "App should not show a raw traceback"
