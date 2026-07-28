"""E2E tests for the AI Analysis page in the Streamlit dashboard.

Covers widget defaults, sidebar interactions, negative/edge-case inputs,
and LLM-driven analysis workflows (graceful when llama-server is down).
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
AI_ANALYSIS_PAGE = "\U0001f916 AI Analysis"

ANALYSIS_TYPE_OPTIONS = [
    "Market Overview",
    "Trade Ideas",
    "Portfolio Review",
    "Risk Assessment",
    "Sector Analysis",
]


@pytest.fixture(autouse=True)
def _navigate_to_ai_analysis(st_page) -> None:
    """Navigate to the AI Analysis page before every test."""
    st_page.navigate_to(AI_ANALYSIS_PAGE)


# -- TestAIAnalysisWidgets ---------------------------------------------------


class TestAIAnalysisWidgets:
    """Verify default widget values and sidebar interactions."""

    def test_page_loads(self, st_page) -> None:
        """AI Analysis page should render without errors."""
        visible = st_page.get_visible_text()
        assert "AI Analysis" in visible

    def test_analysis_type_options(self, st_page) -> None:
        """Analysis Type selectbox should expose exactly five options."""
        options = st_page.get_selectbox_options("Analysis Type", sidebar=True)
        assert options == ANALYSIS_TYPE_OPTIONS

    def test_analysis_type_default(self, st_page) -> None:
        """Default analysis type should be Market Overview."""
        value = st_page.get_selectbox_value("Analysis Type", sidebar=True)
        assert value == "Market Overview"

    def test_temperature_slider_default(self, st_page) -> None:
        """Temperature slider should default to 0.7."""
        value = st_page.get_slider_value("Temperature", sidebar=True)
        assert value == pytest.approx(0.7, abs=0.05)

    def test_max_tokens_slider_default(self, st_page) -> None:
        """Max Tokens slider should default to 1024."""
        value = st_page.get_slider_value("Max Tokens", sidebar=True)
        assert value == 1024

    def test_custom_prompt_textarea(self, st_page) -> None:
        """Custom prompt text area should exist and accept input."""
        st_page.set_text_area("Custom prompt (optional)", "Test prompt text")
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash after typing in text area"


# -- TestAIAnalysisNegative --------------------------------------------------


class TestAIAnalysisNegative:
    """Negative and edge-case tests for error handling."""

    def test_generate_without_llm(self, st_page) -> None:
        """Clicking Generate when llama-server may be down should not crash."""
        st_page.click_button("Generate Analysis", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # Either the LLM produced output or a graceful error message appeared
        assert (
            "LLM unavailable" in visible
            or "Analysis" in visible
            or st_page.has_warning()
        ), "Expected LLM output or a graceful error message"

    def test_temperature_zero(self, st_page) -> None:
        """Setting temperature to 0.0 should not crash the app."""
        st_page.set_slider("Temperature", 0.0, sidebar=True)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash at temperature 0.0"

    def test_temperature_max(self, st_page) -> None:
        """Setting temperature to 1.5 should not crash the app."""
        st_page.set_slider("Temperature", 1.5, sidebar=True)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash at temperature 1.5"


# -- TestAIAnalysisWorkflows -------------------------------------------------


class TestAIAnalysisWorkflows:
    """Full workflow tests for AI analysis generation."""

    def test_generate_with_custom_prompt(self, st_page) -> None:
        """Typing a custom prompt and clicking Generate should produce output or graceful error."""
        st_page.set_text_area(
            "Custom prompt (optional)",
            "What are the top 3 sectors to watch this week?",
        )
        st_page.click_button("Generate Analysis", timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # Accept either real LLM output or a graceful unavailable message
        assert (
            "LLM unavailable" in visible
            or len(visible) > 200
            or st_page.has_warning()
        ), "Expected LLM output or graceful error after custom prompt generation"

    def test_change_analysis_type(self, st_page) -> None:
        """Cycling through all five analysis types should not crash."""
        for analysis_type in ANALYSIS_TYPE_OPTIONS:
            st_page.select_box("Analysis Type", analysis_type, sidebar=True)
            visible = st_page.get_visible_text()
            assert "Traceback" not in visible, (
                f"App should not crash after selecting '{analysis_type}'"
            )
            # The subheader should reflect the selected analysis type
            assert analysis_type in visible, (
                f"Expected '{analysis_type}' in visible text after selection"
            )
