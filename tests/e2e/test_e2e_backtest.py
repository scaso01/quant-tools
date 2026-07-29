"""E2E tests for the Strategy Backtester page in the Streamlit dashboard.

Covers widget defaults, widget interactions, negative/edge-case inputs,
and full backtest workflow execution.
"""

from __future__ import annotations

import pytest

from .streamlit_helpers import (
    SEL_BUTTON,
    SEL_DATE_INPUT,
    SEL_EXPANDER,
    SEL_NUMBER_INPUT,
    SEL_SELECTBOX,
    SEL_SLIDER,
    SEL_TEXT_INPUT,
    StreamlitPage,
)

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
BACKTESTER_PAGE = "\U0001f4c8 Strategy Backtester"

STRATEGY_OPTIONS = ["SMA Crossover", "Bollinger Bands", "RSI", "Momentum"]


@pytest.fixture(autouse=True)
def _navigate_to_backtester(st_page: StreamlitPage) -> None:
    """Navigate to the Strategy Backtester page before every test."""
    st_page.navigate_to(BACKTESTER_PAGE)


# ── TestBacktestWidgets ──────────────────────────────────────────────────


class TestBacktestWidgets:
    """Verify default widget values and basic sidebar interactions."""

    def test_page_loads(self, st_page: StreamlitPage) -> None:
        """Strategy Backtester page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Strategy Backtester" in visible

    def test_ticker_default_aapl(self, st_page: StreamlitPage) -> None:
        """The symbol picker should start the page on AAPL."""
        assert st_page.symbol_picker_count() == 1
        assert "AAPL" in st_page.get_visible_text()

    def test_ticker_change(self, st_page: StreamlitPage) -> None:
        """Choosing MSFT in the symbol picker should drive the page to MSFT."""
        st_page.set_symbol("MSFT")
        assert "MSFT" in st_page.get_visible_text()

    def test_strategy_options(self, st_page: StreamlitPage) -> None:
        """Strategy selectbox should expose exactly four options."""
        options = st_page.get_selectbox_options("Strategy", sidebar=True)
        assert options == STRATEGY_OPTIONS

    def test_strategy_default(self, st_page: StreamlitPage) -> None:
        """Default strategy selection should be SMA Crossover."""
        value = st_page.get_selectbox_value("Strategy", sidebar=True)
        assert value == "SMA Crossover"

    def test_date_range_defaults_to_one_year(
        self, st_page: StreamlitPage
    ) -> None:
        """The date range is chosen with preset buttons, defaulting to 1Y."""
        assert st_page.get_active_date_preset("bt") == "1Y"

    def test_date_range_preset_changes(self, st_page: StreamlitPage) -> None:
        """Clicking a different preset should make it the active range."""
        st_page.click_date_preset("bt", "5Y")
        assert st_page.get_active_date_preset("bt") == "5Y"

    def test_capital_default(self, st_page: StreamlitPage) -> None:
        """Initial capital should default to 100000."""
        value = st_page.get_number_input_value("Initial Capital", sidebar=True)
        assert value == 100_000

    def test_capital_change(self, st_page: StreamlitPage) -> None:
        """Changing initial capital should persist the new value."""
        st_page.set_number_input("Initial Capital", 50000, sidebar=True)
        value = st_page.get_number_input_value("Initial Capital", sidebar=True)
        assert value == 50_000

    def test_commission_slider_default(self, st_page: StreamlitPage) -> None:
        """Commission slider should default to 0.001."""
        value = st_page.get_slider_value("Commission", sidebar=True)
        assert value == pytest.approx(0.001, abs=0.0005)

    def test_strategy_params_expander_opens(self, st_page: StreamlitPage) -> None:
        """Strategy Parameters expander should be openable."""
        st_page.expand_expander("Strategy Parameters", sidebar=True)
        sidebar_text = st_page.get_sidebar_text()
        # After expanding, some parameter labels should be visible
        assert any(
            keyword in sidebar_text
            for keyword in ["Period", "Fast", "Slow", "Std", "Lookback"]
        ), "Expected strategy parameter labels inside the expander"

    def test_sma_params_visible(self, st_page: StreamlitPage) -> None:
        """When SMA Crossover is selected, Fast Period and Slow Period sliders should appear."""
        # SMA Crossover is the default; open the expander
        st_page.expand_expander("Strategy Parameters", sidebar=True)
        sidebar_text = st_page.get_sidebar_text()
        assert "Fast Period" in sidebar_text
        assert "Slow Period" in sidebar_text

    def test_bollinger_params_visible(self, st_page: StreamlitPage) -> None:
        """When Bollinger Bands is selected, Period and Std Deviations sliders should appear."""
        st_page.select_box("Strategy", "Bollinger Bands", sidebar=True)
        st_page.expand_expander("Strategy Parameters", sidebar=True)
        sidebar_text = st_page.get_sidebar_text()
        assert "Period" in sidebar_text
        assert "Std Deviations" in sidebar_text

    def test_run_backtest_button_exists(self, st_page: StreamlitPage) -> None:
        """Run Backtest button should be present in the sidebar."""
        assert st_page.widget_exists(SEL_BUTTON, "Run Backtest", sidebar=True)

    def test_initial_info_message(self, st_page: StreamlitPage) -> None:
        """Before running, an info message prompting the user should be shown."""
        assert st_page.has_info("Configure your backtest")


# ── TestBacktestNegative ─────────────────────────────────────────────────


class TestBacktestNegative:
    """Negative and edge-case tests for error handling."""

    def test_invalid_ticker_offers_no_suggestion(
        self, st_page: StreamlitPage
    ) -> None:
        """The picker only offers real symbols, so a bogus query matches nothing.

        This is what keeps an invalid ticker from reaching the engine at all,
        the old free-text ticker box could submit anything.
        """
        assert st_page.get_symbol_suggestions("ZZZZZ99") == []

    def test_invalid_ticker_does_not_break_a_run(
        self, st_page: StreamlitPage
    ) -> None:
        """A bogus query leaves the page on its default symbol and still runs."""
        st_page.get_symbol_suggestions("ZZZZZ99")
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"

    def test_rsi_oversold_above_overbought(self, st_page: StreamlitPage) -> None:
        """Setting RSI oversold above overbought should be handled gracefully."""
        st_page.select_box("Strategy", "RSI", sidebar=True)
        st_page.expand_expander("Strategy Parameters", sidebar=True)
        # Set oversold high and overbought low (inverted)
        st_page.set_slider("Oversold", 80, sidebar=True)
        st_page.set_slider("Overbought", 30, sidebar=True)
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash on inverted RSI params"

    def test_zero_capital(self, st_page: StreamlitPage) -> None:
        """Minimum capital (1000) should run without crashing."""
        st_page.set_number_input("Initial Capital", 1000, sidebar=True)
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash with minimum capital"

    def test_max_capital(self, st_page: StreamlitPage) -> None:
        """Maximum capital (10000000) should run without crashing."""
        st_page.set_number_input("Initial Capital", 10_000_000, sidebar=True)
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash with maximum capital"


# ── TestBacktestWorkflows ────────────────────────────────────────────────


class TestBacktestWorkflows:
    """Full backtest execution workflows."""

    def test_aapl_sma_crossover(self, st_page: StreamlitPage) -> None:
        """Running AAPL with default SMA Crossover should produce metrics and a chart."""
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)

        # Verify KPI metric cards appear (backtest.py renders 10)
        metric_count = st_page.get_metric_count()
        assert metric_count >= 10, (
            f"Expected at least 10 KPI metrics, got {metric_count}"
        )

        # Verify equity curve Plotly chart
        assert st_page.has_plotly_chart(), (
            "Expected an equity curve Plotly chart after backtest"
        )

    def test_strategy_comparison(self, st_page: StreamlitPage) -> None:
        """After running, a Strategy Comparison expander should be available."""
        st_page.click_button("Run Backtest", sidebar=True, timeout=60_000)

        visible = st_page.get_visible_text()
        assert "Compare All Strategies" in visible, (
            "Expected a Compare All Strategies section after backtest"
        )

    def test_change_strategy_reruns(self, st_page: StreamlitPage) -> None:
        """Selecting Bollinger Bands should update the Strategy Parameters expander."""
        st_page.select_box("Strategy", "Bollinger Bands", sidebar=True)
        st_page.expand_expander("Strategy Parameters", sidebar=True)
        sidebar_text = st_page.get_sidebar_text()

        # Bollinger params should be present, SMA-specific params should not
        assert "Std Deviations" in sidebar_text, (
            "Bollinger Bands params should appear after strategy change"
        )
        assert "Fast Period" not in sidebar_text, (
            "SMA-specific params should not appear for Bollinger Bands"
        )
