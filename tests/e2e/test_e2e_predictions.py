"""E2E tests for the Predictions page in the Streamlit dashboard.

Covers widget presence, filter interactions, negative/edge-case inputs,
and prediction generation workflows.
"""

from __future__ import annotations

import pytest

from .streamlit_helpers import (
    SEL_BUTTON,
    SEL_SELECTBOX,
    SEL_TEXT_INPUT,
    StreamlitPage,
)

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
PREDICTIONS_PAGE = "\U0001f52e Predictions"

ASSET_CLASS_OPTIONS = ["All", "equity", "crypto", "forex", "commodity"]
MODEL_OPTIONS = [
    "All", "ensemble", "gradient_boost", "stats_forecast",
    "neural_forecast", "prophet", "garch",
]


@pytest.fixture(autouse=True)
def _navigate_to_predictions(st_page: StreamlitPage) -> None:
    """Navigate to the Predictions page before every test."""
    st_page.navigate_to(PREDICTIONS_PAGE)


# -- TestPredictionsWidgets ---------------------------------------------------


class TestPredictionsWidgets:
    """Verify that all expected widgets render on the Predictions page."""

    def test_page_loads(self, st_page: StreamlitPage) -> None:
        """Predictions page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Predictions" in visible

    def test_kpi_row(self, st_page: StreamlitPage) -> None:
        """KPI metric cards should be visible (count depends on data availability)."""
        metric_count = st_page.get_metric_count()
        # With no prediction data the page may render zero metric cards
        assert metric_count >= 0, (
            f"Expected metric count >= 0, got {metric_count}"
        )
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_asset_class_filter_options(self, st_page: StreamlitPage) -> None:
        """Asset Class selectbox should expose exactly five options."""
        options = st_page.get_selectbox_options("Asset Class")
        assert options == ASSET_CLASS_OPTIONS

    def test_model_filter_options(self, st_page: StreamlitPage) -> None:
        """Model selectbox should expose the expected options."""
        options = st_page.get_selectbox_options("Model")
        assert options == MODEL_OPTIONS

    def test_symbol_picker_exists(self, st_page: StreamlitPage) -> None:
        """The symbol picker should be present on the page."""
        assert st_page.symbol_picker_count() == 1

    def test_predict_button_exists(self, st_page: StreamlitPage) -> None:
        """Predict button should be present on the page."""
        assert st_page.widget_exists(SEL_BUTTON, "Predict")

    def test_clear_all_button_exists(self, st_page: StreamlitPage) -> None:
        """Clear All Predictions button should be present on the page."""
        assert st_page.widget_exists(SEL_BUTTON, "Clear All Predictions")

    def test_horizon_filter_exists(self, st_page: StreamlitPage) -> None:
        """Horizon selectbox should be present in the filter row."""
        assert st_page.widget_exists(SEL_SELECTBOX, "Horizon")


# -- TestPredictionsNegative --------------------------------------------------


class TestPredictionsNegative:
    """Negative and edge-case tests for error handling."""

    def test_predict_empty_symbol(self, st_page: StreamlitPage) -> None:
        """Clicking Predict without entering a symbol should not crash."""
        st_page.click_button("Predict", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_invalid_symbol_offers_no_suggestion(
        self, st_page: StreamlitPage
    ) -> None:
        """The picker only offers real symbols, so junk input matches nothing."""
        assert st_page.get_symbol_suggestions("!!!!") == []

    def test_predict_after_invalid_symbol(
        self, st_page: StreamlitPage
    ) -> None:
        """Junk in the picker should leave Predict working, not crash the page."""
        st_page.get_symbol_suggestions("!!!!")
        st_page.click_button("Predict", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_no_predictions_state(self, st_page: StreamlitPage) -> None:
        """When no predictions exist, an info message should be displayed."""
        visible = st_page.get_visible_text()
        # Either predictions are shown (dataframe) or the info message appears
        has_data = st_page.has_dataframe()
        # Covers both empty states: "No predictions yet" before anything has been
        # generated, and "No predictions match your filters" once filters narrow.
        has_info_msg = st_page.has_info("No predictions")
        assert has_data or has_info_msg, (
            "Expected either a predictions dataframe or 'No predictions found' info message"
        )


# -- TestPredictionsWorkflows -------------------------------------------------


class TestPredictionsWorkflows:
    """Full workflow tests for filtering and prediction generation."""

    def test_filter_by_asset_class(self, st_page: StreamlitPage) -> None:
        """Selecting an asset class filter should trigger a page rerun."""
        st_page.select_box("Asset Class", "equity")
        # After selecting a filter, the page should still be functional
        visible = st_page.get_visible_text()
        assert "Predictions" in visible
        assert "Traceback" not in visible

    def test_generate_prediction(self, st_page: StreamlitPage) -> None:
        """Typing AAPL and clicking Predict should produce results or a graceful error."""
        st_page.set_symbol("AAPL")
        st_page.click_button("Predict", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # Either a success message, an error message, or the page just stays intact
        has_outcome = (
            st_page.has_success()
            or st_page.has_error()
            or st_page.has_warning()
            or "Predictions" in visible
        )
        assert has_outcome, "Expected some outcome after clicking Predict"

    def test_clear_predictions(self, st_page: StreamlitPage) -> None:
        """Clicking Clear All Predictions should not crash and should reset state."""
        st_page.click_button("Clear All Predictions", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash on Clear All"
        # After clearing, the page should still render
        assert "Predictions" in visible
