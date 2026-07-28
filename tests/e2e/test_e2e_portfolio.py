"""E2E Playwright tests for the Portfolio Analyzer page.

Covers widget rendering, preset buttons, input validation (negative tests),
and full analysis workflows including Monte Carlo distribution changes.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tests.e2e.streamlit_helpers import (
    SEL_BUTTON,
    SEL_FILE_UPLOADER,
    StreamlitPage,
)

pytestmark = [pytest.mark.e2e]

# ── Constants ────────────────────────────────────────────────────────────

PAGE_NAME = "\U0001f4bc Portfolio Analyzer"

DEFAULT_HOLDINGS_JSON = '{"AAPL": 0.3, "MSFT": 0.2, "BND": 0.3, "GLD": 0.2}'

PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y"]

DISTRIBUTION_OPTIONS = [
    "Normal",
    "Student-t",
    "Skew-Normal",
    "Johnson SU",
    "GED",
    "Auto-detect (BIC)",
]

ANALYSIS_TIMEOUT = 120_000


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _navigate_to_portfolio(st_page: StreamlitPage) -> None:
    """Navigate to the Portfolio Analyzer page before every test."""
    st_page.navigate_to(PAGE_NAME)


# ── TestPortfolioWidgets ─────────────────────────────────────────────────


class TestPortfolioWidgets:
    """Verify all sidebar widgets render correctly on initial load."""

    def test_page_loads(self, st_page: StreamlitPage) -> None:
        """Portfolio Analyzer heading should be visible after navigation."""
        visible = st_page.get_visible_text()
        assert "Portfolio Analyzer" in visible

    def test_preset_60_40_button(self, st_page: StreamlitPage) -> None:
        """The '60/40' preset button should exist in the sidebar."""
        scope = st_page._scope(sidebar=True)
        assert scope.locator('button:has-text("60/40")').count() > 0, (
            "60/40 preset button not found in sidebar"
        )

    def test_preset_tech_heavy_button(self, st_page: StreamlitPage) -> None:
        """The 'Tech Heavy' preset button should exist in the sidebar."""
        scope = st_page._scope(sidebar=True)
        assert scope.locator('button:has-text("Tech Heavy")').count() > 0, (
            "Tech Heavy preset button not found in sidebar"
        )

    def test_preset_all_weather_button(self, st_page: StreamlitPage) -> None:
        """The 'All-Weather' preset button should exist in the sidebar."""
        scope = st_page._scope(sidebar=True)
        assert scope.locator('button:has-text("All-Weather")').count() > 0, (
            "All-Weather preset button not found in sidebar"
        )

    def test_holdings_textarea_default(self, st_page: StreamlitPage) -> None:
        """Holdings textarea should contain the default JSON on first load."""
        value = st_page.get_text_area_value("Holdings (JSON)", sidebar=True)
        assert "AAPL" in value
        assert "MSFT" in value
        assert "BND" in value
        assert "GLD" in value

    def test_holdings_textarea_edit(self, st_page: StreamlitPage) -> None:
        """Editing the holdings textarea should persist the new value."""
        new_holdings = '{"SPY": 0.5, "BND": 0.5}'
        st_page.set_text_area("Holdings (JSON)", new_holdings, sidebar=True)
        value = st_page.get_text_area_value("Holdings (JSON)", sidebar=True)
        assert "SPY" in value
        assert "BND" in value

    def test_period_selectbox_options(self, st_page: StreamlitPage) -> None:
        """Period selectbox should offer exactly 4 options."""
        options = st_page.get_selectbox_options("Period", sidebar=True)
        assert len(options) == 4, f"Expected 4 period options, got {len(options)}: {options}"
        for expected in PERIOD_OPTIONS:
            assert any(expected in opt for opt in options), (
                f"Period option '{expected}' not found in {options}"
            )

    def test_risk_free_rate_slider_default(self, st_page: StreamlitPage) -> None:
        """Risk-Free Rate slider should default to 0.04."""
        value = st_page.get_slider_value("Risk-Free Rate", sidebar=True)
        assert abs(value - 0.04) < 0.01, f"Expected ~0.04, got {value}"

    def test_benchmark_selectbox_exists(self, st_page: StreamlitPage) -> None:
        """Benchmark selectbox should be present in the sidebar."""
        sidebar_text = st_page.get_sidebar_text()
        assert "Benchmark" in sidebar_text, (
            "Benchmark selectbox not found in sidebar"
        )

    def test_file_uploader_hidden_by_default(self, st_page: StreamlitPage) -> None:
        """File uploader should NOT be visible when benchmark is not 'Custom Upload'."""
        scope = st_page._scope(sidebar=True)
        uploaders = scope.locator(SEL_FILE_UPLOADER)
        assert uploaders.count() == 0, (
            "File uploader should be hidden when benchmark is not Custom Upload"
        )

    def test_monte_carlo_sims_slider_default(self, st_page: StreamlitPage) -> None:
        """Monte Carlo Simulations slider should default to 1000."""
        value = st_page.get_slider_value("Simulations", sidebar=True)
        assert value == 1000, f"Expected 1000 simulations, got {value}"

    def test_monte_carlo_horizon_default(self, st_page: StreamlitPage) -> None:
        """Monte Carlo Trading Days slider should default to 252."""
        value = st_page.get_slider_value("Trading Days", sidebar=True)
        assert value == 252, f"Expected 252 trading days, got {value}"

    def test_distribution_selectbox_options(self, st_page: StreamlitPage) -> None:
        """Distribution selectbox should offer exactly 6 options."""
        options = st_page.get_selectbox_options("Distribution", sidebar=True)
        assert len(options) == 6, (
            f"Expected 6 distribution options, got {len(options)}: {options}"
        )

    def test_analyze_button_exists(self, st_page: StreamlitPage) -> None:
        """The 'Analyze Portfolio' primary button should be in the sidebar."""
        assert st_page.widget_exists(SEL_BUTTON, "Analyze Portfolio", sidebar=True), (
            "Analyze Portfolio button not found in sidebar"
        )

    def test_initial_state(self, st_page: StreamlitPage) -> None:
        """Before clicking Analyze, the Portfolio Analyzer page should be visible."""
        visible = st_page.get_visible_text()
        sidebar = st_page.get_sidebar_text()
        # The page header or sidebar should contain Portfolio-related content
        has_portfolio_heading = "Portfolio Analyzer" in visible
        has_portfolio_sidebar = "Holdings (JSON)" in sidebar or "Portfolio Settings" in sidebar
        has_info_prompt = "Enter your holdings" in visible or "Analyze Portfolio" in visible
        assert has_portfolio_heading or has_portfolio_sidebar or has_info_prompt, (
            "Initial state should show the Portfolio Analyzer page"
        )


# ── TestPortfolioNegative ────────────────────────────────────────────────


class TestPortfolioNegative:
    """Validation and error-handling tests for bad inputs."""

    def test_invalid_json(self, st_page: StreamlitPage) -> None:
        """Submitting malformed JSON should show an error, not crash."""
        st_page.set_text_area("Holdings (JSON)", "{invalid}", sidebar=True)
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App crashed with unhandled exception"
        # App should show some kind of error or not crash
        has_json_error = st_page.has_error("Invalid JSON") or st_page.has_error("invalid")
        has_any_indicator = has_json_error or st_page.has_error() or "error" in visible.lower()
        assert has_any_indicator or "Enter your holdings" in visible, (
            "Expected an error message or safe state for invalid JSON input"
        )

    def test_empty_json_object(self, st_page: StreamlitPage) -> None:
        """Submitting an empty JSON object '{}' should show an error."""
        st_page.set_text_area("Holdings (JSON)", "{}", sidebar=True)
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App crashed with unhandled exception"
        has_empty_error = st_page.has_error("non-empty") or st_page.has_error("Holdings must")
        has_any_indicator = has_empty_error or st_page.has_error() or "error" in visible.lower()
        assert has_any_indicator or "Enter your holdings" in visible, (
            "Expected an error message or safe state for empty JSON object"
        )

    def test_nonexistent_ticker(self, st_page: StreamlitPage) -> None:
        """A nonexistent ticker should produce a data error, not a crash."""
        st_page.set_text_area(
            "Holdings (JSON)", '{"ZZZZZ99": 1.0}', sidebar=True
        )
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=120_000)

        visible = st_page.get_visible_text()
        # Must not show a Python traceback (unhandled exception)
        assert "Traceback" not in visible, "App crashed with unhandled exception"

    def test_single_holding(self, st_page: StreamlitPage) -> None:
        """A single-holding portfolio should not crash (edge case for correlation)."""
        st_page.set_text_area(
            "Holdings (JSON)", '{"AAPL": 1.0}', sidebar=True
        )
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "Single-holding portfolio caused a crash"
        )

    def test_negative_weights(self, st_page: StreamlitPage) -> None:
        """Negative weights (short positions) should be handled gracefully."""
        st_page.set_text_area(
            "Holdings (JSON)", '{"AAPL": -0.5, "MSFT": 1.5}', sidebar=True
        )
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "Negative weights caused an unhandled crash"
        )

    def test_empty_textarea(self, st_page: StreamlitPage) -> None:
        """Clearing the textarea entirely should show an error."""
        st_page.set_text_area("Holdings (JSON)", " ", sidebar=True)
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App crashed with unhandled exception"
        has_json_error = st_page.has_error("Invalid JSON") or st_page.has_error("invalid")
        has_empty_error = st_page.has_error("non-empty") or st_page.has_error("Holdings must")
        has_any_indicator = has_json_error or has_empty_error or st_page.has_error() or "error" in visible.lower()
        assert has_any_indicator or "Enter your holdings" in visible, (
            "Expected an error message or safe state for empty textarea"
        )


# ── TestPortfolioWorkflows ───────────────────────────────────────────────


class TestPortfolioWorkflows:
    """End-to-end analysis workflows that exercise the full pipeline."""

    def test_default_portfolio_analysis(self, st_page: StreamlitPage) -> None:
        """Running analysis with default holdings should produce metrics and charts."""
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "Analysis crashed"

        # Should render KPI metrics (Total Return, Sharpe, etc.)
        metric_count = st_page.get_metric_count()
        assert metric_count >= 4, (
            f"Expected at least 4 metrics after analysis, got {metric_count}"
        )

        # Should render at least one Plotly chart (sector donut, correlation, MC fan)
        assert st_page.has_plotly_chart(), "Expected at least one Plotly chart"

        # Should render the holdings dataframe
        assert st_page.has_dataframe(), "Expected the holdings dataframe"

    def test_preset_then_analyze(self, st_page: StreamlitPage) -> None:
        """Clicking a preset then Analyze should work with preset tickers."""
        st_page.click_button("60/40", sidebar=True)

        # Verify the textarea was updated with preset holdings
        value = st_page.get_text_area_value("Holdings (JSON)", sidebar=True)
        assert "SPY" in value, "60/40 preset should contain SPY"
        assert "BND" in value, "60/40 preset should contain BND"

        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "Analysis crashed after 60/40 preset"
        assert st_page.get_metric_count() >= 4, "Expected metrics after preset analysis"

    def test_change_period_and_reanalyze(self, st_page: StreamlitPage) -> None:
        """Changing period to 3mo and analyzing should still produce results."""
        st_page.select_box("Period", "3mo", sidebar=True)
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "Reanalysis with 3mo period crashed"
        assert st_page.get_metric_count() >= 4, (
            "Expected metrics after 3mo period analysis"
        )

    def test_monte_carlo_distribution_change(self, st_page: StreamlitPage) -> None:
        """Changing Monte Carlo distribution to Student-t should not crash."""
        st_page.select_box("Distribution", "Student-t", sidebar=True)
        st_page.click_button("Analyze Portfolio", sidebar=True, timeout=ANALYSIS_TIMEOUT)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "Analysis crashed with Student-t distribution"
        )
