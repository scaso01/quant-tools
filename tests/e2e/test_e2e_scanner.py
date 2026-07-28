"""E2E Playwright tests for the Market Scanner page.

Tests widget rendering, default states, user interactions, validation
messages, and full scan workflows across all four asset classes.
"""

from __future__ import annotations

import pytest

from .streamlit_helpers import SEL_SELECTBOX

pytestmark = [pytest.mark.e2e]

# ── Helpers ──────────────────────────────────────────────────────────

SCANNER_PAGE = "\U0001f4e1 Market Scanner"

SIGNAL_LABELS = ["Momentum", "Value", "Quality", "Mean Rev."]

EQUITY_UNIVERSES = [
    "S&P 500 Top 100",
    "S&P 500",
    "Russell 1000",
    "International EAFE",
    "Emerging Markets",
]


def _navigate(st_page) -> None:
    """Navigate to the Market Scanner page and wait for it to settle."""
    st_page.navigate_to(SCANNER_PAGE)


def _uncheck_all_signals(st_page) -> None:
    """Uncheck all four signal checkboxes in the sidebar."""
    for label in SIGNAL_LABELS:
        st_page.check_box(label, checked=False, sidebar=True)


# ── Widget Rendering & Defaults ──────────────────────────────────────


class TestScannerWidgets:
    """Verify sidebar widgets render with correct defaults."""

    def test_page_loads(self, st_page) -> None:
        _navigate(st_page)
        text = st_page.get_visible_text()
        assert "Market Scanner" in text

    def test_asset_class_selectbox_options(self, st_page) -> None:
        _navigate(st_page)
        options = st_page.get_selectbox_options("Asset Class", sidebar=True)
        assert options == ["Equity", "Crypto", "Forex", "Commodity"]

    def test_asset_class_default_is_equity(self, st_page) -> None:
        _navigate(st_page)
        value = st_page.get_selectbox_value("Asset Class", sidebar=True)
        assert value == "Equity"

    def test_universe_selectbox_has_presets(self, st_page) -> None:
        _navigate(st_page)
        options = st_page.get_selectbox_options("Universe", sidebar=True)
        for preset in EQUITY_UNIVERSES:
            assert preset in options, f"Missing universe preset: {preset}"

    def test_universe_changes_with_asset_class(self, st_page) -> None:
        _navigate(st_page)
        # Get sidebar text before and after asset class change
        st_page.select_box("Asset Class", "Equity", sidebar=True)
        equity_text = st_page.get_sidebar_text()
        st_page.select_box("Asset Class", "Crypto", sidebar=True)
        crypto_text = st_page.get_sidebar_text()
        # Sidebar content should change when asset class changes
        assert equity_text != crypto_text

    def test_commodity_category_appears(self, st_page) -> None:
        _navigate(st_page)
        st_page.select_box("Asset Class", "Commodity", sidebar=True)
        assert st_page.widget_exists(
            SEL_SELECTBOX, "Commodity Category", sidebar=True
        )

    def test_signal_checkboxes_default_checked(self, st_page) -> None:
        _navigate(st_page)
        for label in SIGNAL_LABELS:
            assert st_page.is_checked(label, sidebar=True), (
                f"{label} should be checked by default"
            )

    def test_signal_checkbox_toggle(self, st_page) -> None:
        _navigate(st_page)
        st_page.check_box("Momentum", checked=False, sidebar=True)
        assert not st_page.is_checked("Momentum", sidebar=True)

    def test_top_n_slider_default(self, st_page) -> None:
        _navigate(st_page)
        value = st_page.get_slider_value("Top N", sidebar=True)
        assert value == 20

    def test_top_n_slider_change(self, st_page) -> None:
        _navigate(st_page)
        st_page.set_slider("Top N", 10, sidebar=True)
        value = st_page.get_slider_value("Top N", sidebar=True)
        assert value == 10

    def test_timeframe_selectbox_exists(self, st_page) -> None:
        _navigate(st_page)
        assert st_page.widget_exists(
            SEL_SELECTBOX, "Timeframe", sidebar=True
        )

    def test_sector_filter_expander_opens(self, st_page) -> None:
        _navigate(st_page)
        st_page.expand_expander("Sector Filter", sidebar=True)
        sidebar_text = st_page.get_sidebar_text()
        # After expanding, the All / None buttons should be visible
        assert "All" in sidebar_text or "None" in sidebar_text

    def test_run_scan_button_exists(self, st_page) -> None:
        _navigate(st_page)
        scope = st_page._scope(sidebar=True)
        btn = scope.locator('button:has-text("Run Scan")')
        assert btn.count() > 0

    def test_initial_info_message(self, st_page) -> None:
        _navigate(st_page)
        assert st_page.has_info("Configure scanner settings")


# ── Negative / Validation Tests ──────────────────────────────────────


class TestScannerNegative:
    """Verify validation messages and error handling."""

    def test_no_signals_shows_warning(self, st_page) -> None:
        _navigate(st_page)
        _uncheck_all_signals(st_page)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        assert st_page.has_warning("Select at least one signal type")

    def test_large_universe_equity_scan(self, st_page) -> None:
        """Motivating bug test: largest equity universe must not crash.

        Selects the largest available equity universe and runs a full
        scan.  The test passes if results appear OR if a graceful
        warning is shown.  It fails only if a Python traceback /
        unhandled exception is rendered.
        """
        _navigate(st_page)
        st_page.select_box("Asset Class", "Equity", sidebar=True)
        st_page.select_box("Universe", "Russell 1000", sidebar=True)
        for label in SIGNAL_LABELS:
            st_page.check_box(label, checked=True, sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=120_000)

        visible = st_page.get_visible_text()
        has_results = st_page.has_dataframe() or st_page.has_plotly_chart()
        has_graceful_msg = (
            st_page.has_warning() or st_page.has_info() or st_page.has_error()
        )
        has_traceback = "Traceback" in visible or "Exception" in visible

        assert not has_traceback, (
            "Python traceback visible on page after Global All scan"
        )
        assert has_results or has_graceful_msg, (
            "Neither results nor a user-facing message appeared"
        )

    def test_sector_filter_none_clicked(self, st_page) -> None:
        _navigate(st_page)
        st_page.expand_expander("Sector Filter", sidebar=True)
        st_page.click_button("None", sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        assert st_page.has_warning("No symbols match")


# ── Full Scan Workflows ─────────────────────────────────────────────


class TestScannerWorkflows:
    """End-to-end scan workflows for each asset class."""

    @staticmethod
    def _assert_scan_ok(st_page) -> None:
        """After clicking Run Scan, verify results or a graceful message."""
        visible = st_page.get_visible_text()
        has_results = st_page.has_dataframe() or st_page.has_plotly_chart()
        has_graceful = (
            st_page.has_warning() or st_page.has_info() or st_page.has_error()
        )
        has_traceback = "Traceback" in visible or "Exception" in visible

        assert not has_traceback, "Python traceback visible on page"
        assert has_results or has_graceful, (
            "Neither results nor a user-facing message appeared"
        )

    def test_equity_sp500_full_scan(self, st_page) -> None:
        _navigate(st_page)
        st_page.select_box("Asset Class", "Equity", sidebar=True)
        st_page.select_box("Universe", "S&P 500", sidebar=True)
        for label in SIGNAL_LABELS:
            st_page.check_box(label, checked=True, sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        self._assert_scan_ok(st_page)

    def test_crypto_scan(self, st_page) -> None:
        _navigate(st_page)
        st_page.select_box("Asset Class", "Crypto", sidebar=True)
        for label in SIGNAL_LABELS:
            st_page.check_box(label, checked=True, sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        self._assert_scan_ok(st_page)

    def test_forex_scan(self, st_page) -> None:
        _navigate(st_page)
        st_page.select_box("Asset Class", "Forex", sidebar=True)
        for label in SIGNAL_LABELS:
            st_page.check_box(label, checked=True, sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        self._assert_scan_ok(st_page)

    def test_commodity_scan(self, st_page) -> None:
        _navigate(st_page)
        st_page.select_box("Asset Class", "Commodity", sidebar=True)
        for label in SIGNAL_LABELS:
            st_page.check_box(label, checked=True, sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)
        self._assert_scan_ok(st_page)

    def test_scan_then_change_asset_class(self, st_page) -> None:
        """Run an equity scan, then switch to crypto without crashing."""
        _navigate(st_page)
        st_page.select_box("Asset Class", "Equity", sidebar=True)
        st_page.click_button("Run Scan", sidebar=True, timeout=60_000)

        # Now switch asset class — page should not error
        st_page.select_box("Asset Class", "Crypto", sidebar=True)

        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, (
            "Page crashed after switching asset class post-scan"
        )
