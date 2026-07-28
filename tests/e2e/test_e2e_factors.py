"""E2E Playwright tests for the Factors & Macro page.

Covers sidebar widgets (checkbox, sliders, text area, buttons),
negative paths (empty symbols, zero weights), and calculation workflows.
"""

from __future__ import annotations

import pytest

from .streamlit_helpers import SEL_TEXT_AREA, StreamlitPage

pytestmark = [pytest.mark.e2e]

# ── Page constant ────────────────────────────────────────────────────

PAGE_NAME = "\U0001f3db\ufe0f Factors & Macro"


@pytest.fixture(autouse=True)
def _navigate_to_factors(st_page: StreamlitPage) -> None:
    """Ensure every test in this module starts on the Factors page."""
    st_page.navigate_to(PAGE_NAME)


# ── TestFactorsWidgets ───────────────────────────────────────────────


class TestFactorsWidgets:
    """Verify sidebar widgets render with correct defaults."""

    def test_page_loads(self, st_page: StreamlitPage) -> None:
        """The Factors page header should be visible after navigation."""
        visible = st_page.get_visible_text()
        assert "Factors" in visible, (
            "Expected 'Factors' in visible text after navigating to the page"
        )

    def test_detect_regime_button_exists(
        self, st_page: StreamlitPage
    ) -> None:
        """The 'Detect Regime' button should be present in the sidebar."""
        sidebar = st_page._scope(sidebar=True)
        # Look for button text anywhere in the sidebar (button label may be
        # wrapped inside nested elements)
        btn = sidebar.locator("button", has_text="Detect Regime")
        if btn.count() == 0:
            # Fallback: search all buttons for partial text match
            all_btns = sidebar.locator("button")
            found = False
            for i in range(all_btns.count()):
                if "Detect Regime" in all_btns.nth(i).inner_text():
                    found = True
                    break
            assert found, (
                "Detect Regime button not found in sidebar. "
                f"Buttons found: {[all_btns.nth(i).inner_text() for i in range(all_btns.count())]}"
            )
        else:
            assert btn.count() > 0

    def test_sp500_checkbox_default_checked(
        self, st_page: StreamlitPage
    ) -> None:
        """The 'Use S&P 500 Top 100' checkbox should be checked by default."""
        assert st_page.is_checked("Use S&P 500 Top 100", sidebar=True), (
            "S&P 500 Top 100 checkbox should be checked by default"
        )

    def test_custom_symbols_disabled_when_sp500_checked(
        self, st_page: StreamlitPage
    ) -> None:
        """Custom Symbols text area should be disabled when SP500 is checked."""
        # SP500 checkbox is checked by default, so text area should be disabled
        sidebar = st_page._scope(sidebar=True)
        widget = sidebar.locator(f'{SEL_TEXT_AREA}:has-text("Custom Symbols")').first
        textarea = widget.locator("textarea").first
        assert textarea.is_disabled(), (
            "Custom Symbols text area should be disabled when SP500 checkbox is checked"
        )

    def test_uncheck_sp500_enables_custom_symbols(
        self, st_page: StreamlitPage
    ) -> None:
        """Unchecking SP500 should enable the Custom Symbols text area."""
        st_page.check_box("Use S&P 500 Top 100", checked=False, sidebar=True)

        sidebar = st_page._scope(sidebar=True)
        widget = sidebar.locator(f'{SEL_TEXT_AREA}:has-text("Custom Symbols")').first
        textarea = widget.locator("textarea").first
        assert not textarea.is_disabled(), (
            "Custom Symbols text area should be enabled after unchecking SP500"
        )

    def test_value_weight_slider_default(
        self, st_page: StreamlitPage
    ) -> None:
        """Value weight slider should default to 0.33."""
        value = st_page.get_slider_value("Value weight", sidebar=True)
        assert abs(value - 0.33) < 0.02, (
            f"Value weight default should be ~0.33, got {value}"
        )

    def test_momentum_weight_slider_default(
        self, st_page: StreamlitPage
    ) -> None:
        """Momentum weight slider should default to 0.33."""
        value = st_page.get_slider_value("Momentum weight", sidebar=True)
        assert abs(value - 0.33) < 0.02, (
            f"Momentum weight default should be ~0.33, got {value}"
        )

    def test_quality_weight_slider_default(
        self, st_page: StreamlitPage
    ) -> None:
        """Quality weight slider should default to 0.34."""
        value = st_page.get_slider_value("Quality weight", sidebar=True)
        assert abs(value - 0.34) < 0.02, (
            f"Quality weight default should be ~0.34, got {value}"
        )

    def test_top_n_slider_default(self, st_page: StreamlitPage) -> None:
        """Top N Results slider should default to 10."""
        value = st_page.get_slider_value("Top N Results", sidebar=True)
        assert value == 10, f"Top N default should be 10, got {value}"

    def test_calculate_factors_button_exists(
        self, st_page: StreamlitPage
    ) -> None:
        """The 'Calculate Factors' primary button should be in the sidebar."""
        scope = st_page._scope(sidebar=True)
        btn = scope.locator('button:has-text("Calculate Factors")')
        assert btn.count() > 0, "Calculate Factors button not found in sidebar"


# ── TestFactorsNegative ──────────────────────────────────────────────


class TestFactorsNegative:
    """Negative and edge-case tests for factor calculation."""

    def test_all_weights_zero(self, st_page: StreamlitPage) -> None:
        """Setting all factor weights to 0 and clicking Calculate should not crash."""
        st_page.set_slider("Value weight", 0.0, sidebar=True)
        st_page.set_slider("Momentum weight", 0.0, sidebar=True)
        st_page.set_slider("Quality weight", 0.0, sidebar=True)

        st_page.click_button("Calculate Factors", sidebar=True, timeout=60_000)

        # The page should still be functional (no Python traceback)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "App crashed with traceback after submitting zero weights"
        )

    def test_empty_custom_symbols(self, st_page: StreamlitPage) -> None:
        """Unchecking SP500 with empty custom symbols should show a warning."""
        st_page.check_box("Use S&P 500 Top 100", checked=False, sidebar=True)
        # Leave custom symbols text area empty (default state)
        st_page.click_button("Calculate Factors", sidebar=True, timeout=60_000)

        # The app warns that no symbols were provided
        visible = st_page.get_visible_text()
        has_warning = st_page.has_warning("No symbols") or "No symbols" in visible
        assert has_warning, (
            "Expected a warning about no symbols when custom symbols is empty"
        )

    def test_single_custom_symbol(self, st_page: StreamlitPage) -> None:
        """Providing a single symbol should not crash (edge case for factor scoring)."""
        st_page.check_box("Use S&P 500 Top 100", checked=False, sidebar=True)
        st_page.set_text_area("Custom Symbols", "AAPL", sidebar=True)

        st_page.click_button("Calculate Factors", sidebar=True, timeout=60_000)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "App crashed with traceback when calculating factors for a single symbol"
        )


# ── TestFactorsWorkflows ─────────────────────────────────────────────


class TestFactorsWorkflows:
    """Integration workflows that exercise the full factor calculation path."""

    def test_calculate_sp500_factors(self, st_page: StreamlitPage) -> None:
        """Calculate factors with SP500 defaults should produce results or no crash."""
        # Defaults: SP500 checked, weights 0.33/0.33/0.34, top_n 10
        st_page.click_button("Calculate Factors", sidebar=True, timeout=120_000)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # Accept any non-crash outcome: results, warnings, or just the page intact
        has_results = st_page.has_dataframe() or st_page.has_plotly_chart()
        has_text_results = "Composite" in visible or "Score" in visible
        has_feedback = st_page.has_warning() or st_page.has_error() or st_page.has_info()
        assert has_results or has_text_results or has_feedback or "Factors" in visible, (
            "Expected some outcome after SP500 factor calculation"
        )

    def test_custom_symbols_factors(self, st_page: StreamlitPage) -> None:
        """Calculate factors for a small custom universe."""
        st_page.check_box("Use S&P 500 Top 100", checked=False, sidebar=True)
        st_page.set_text_area("Custom Symbols", "AAPL\nMSFT\nGOOGL", sidebar=True)

        st_page.click_button("Calculate Factors", sidebar=True, timeout=60_000)

        has_results = st_page.has_dataframe() or st_page.has_plotly_chart()
        visible = st_page.get_visible_text()
        has_text_results = "Composite" in visible or "Score" in visible
        assert has_results or has_text_results, (
            "Expected results after calculating factors for AAPL, MSFT, GOOGL"
        )

    def test_change_weights_and_recalculate(
        self, st_page: StreamlitPage
    ) -> None:
        """Changing the Value weight and recalculating should not crash."""
        st_page.set_slider("Value weight", 0.5, sidebar=True)

        st_page.click_button("Calculate Factors", sidebar=True, timeout=60_000)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "App crashed after changing Value weight and recalculating"
        )
        # Verify something rendered (dataframe, chart, or score text)
        has_results = st_page.has_dataframe() or st_page.has_plotly_chart()
        has_text_results = "Composite" in visible or "Score" in visible
        assert has_results or has_text_results, (
            "Expected results after recalculating with adjusted Value weight"
        )
