"""Textual TUI dashboard (deco-monitor pattern)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header

from quant_tools.models import (
    DashboardState,
    DashboardTab,
)
from quant_tools.tui.widgets import (
    BacktestPanel,
    FactorsPanel,
    PortfolioPanel,
    ScannerPanel,
    StatusPanel,
)

log = logging.getLogger("quant_tools")


class QuantToolsApp(App):
    """Quant-Tools TUI Dashboard."""

    TITLE = "Quant-Tools"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main-scroll {
        height: 1fr;
    }
    #scanner-panel {
        height: auto;
        min-height: 10;
        border: round $primary;
        padding: 0 1;
    }
    #backtest-panel {
        height: auto;
        min-height: 8;
        border: round $accent;
        padding: 0 1;
    }
    #portfolio-panel {
        height: auto;
        min-height: 10;
        border: round $primary;
        padding: 0 1;
    }
    #factors-panel {
        height: auto;
        min-height: 10;
        border: round $secondary;
        padding: 0 1;
    }
    #status-panel {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("1", "tab_scanner", "Scanner"),
        Binding("2", "tab_backtest", "Backtest"),
        Binding("3", "tab_portfolio", "Portfolio"),
        Binding("4", "tab_factors", "Factors"),
        Binding("r", "refresh", "Refresh"),
    ]

    active_tab: reactive[DashboardTab] = reactive(DashboardTab.SCANNER)

    def __init__(self) -> None:
        super().__init__()
        self._state = DashboardState()

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="main-scroll"):
            yield ScannerPanel(id="scanner-panel")
            yield BacktestPanel(id="backtest-panel")
            yield PortfolioPanel(id="portfolio-panel")
            yield FactorsPanel(id="factors-panel")

        yield StatusPanel(id="status-panel")
        yield Footer()

    def on_mount(self) -> None:
        self._update_tab_visibility()
        self._update_ui()

    def _update_tab_visibility(self) -> None:
        """Show only the active tab's panel."""
        tab_panels = {
            DashboardTab.SCANNER: "scanner-panel",
            DashboardTab.BACKTEST: "backtest-panel",
            DashboardTab.PORTFOLIO: "portfolio-panel",
            DashboardTab.FACTORS: "factors-panel",
        }
        for tab, panel_id in tab_panels.items():
            self.query_one(f"#{panel_id}").display = (tab == self._state.active_tab)

    def _update_ui(self) -> None:
        """Refresh all visible panels."""
        self.query_one("#scanner-panel", ScannerPanel).update_data(
            self._state.scan_results
        )
        self.query_one("#backtest-panel", BacktestPanel).update_data(
            self._state.backtest_result
        )
        self.query_one("#portfolio-panel", PortfolioPanel).update_data(
            self._state.portfolio_analysis
        )
        self.query_one("#factors-panel", FactorsPanel).update_data(
            self._state.factor_scores, self._state.macro_snapshot
        )
        self.query_one("#status-panel", StatusPanel).update_data(self._state)

    def watch_active_tab(self, value: DashboardTab) -> None:
        self._state.active_tab = value
        self._update_tab_visibility()
        self._update_ui()

    def action_tab_scanner(self) -> None:
        self.active_tab = DashboardTab.SCANNER

    def action_tab_backtest(self) -> None:
        self.active_tab = DashboardTab.BACKTEST

    def action_tab_portfolio(self) -> None:
        self.active_tab = DashboardTab.PORTFOLIO

    def action_tab_factors(self) -> None:
        self.active_tab = DashboardTab.FACTORS

    @work(exclusive=True, thread=True)
    def action_refresh(self) -> None:
        """Refresh data for the active tab."""
        self._state.is_loading = True
        self._state.status_message = "Refreshing..."
        self.call_from_thread(self._update_ui)

        try:
            if self._state.active_tab == DashboardTab.SCANNER:
                self._refresh_scanner()
            elif self._state.active_tab == DashboardTab.BACKTEST:
                self._state.status_message = "Use CLI for backtest: quant-tools backtest"
            elif self._state.active_tab == DashboardTab.PORTFOLIO:
                self._state.status_message = "Use CLI for portfolio: quant-tools portfolio"
            elif self._state.active_tab == DashboardTab.FACTORS:
                self._refresh_factors()
        except Exception:
            log.exception("Refresh failed")
            self._state.status_message = "Refresh failed"
        finally:
            self._state.is_loading = False
            self._state.last_update = datetime.now()
            self.call_from_thread(self._update_ui)

    def _refresh_scanner(self) -> None:
        """Refresh scanner data in background thread."""
        try:
            from quant_tools.scanner.engine import scan_universe
            from quant_tools.data.universe import get_universe

            symbols = get_universe("sp500")[:30]  # Top 30 for speed
            results = scan_universe(symbols, top_n=20)
            self._state.scan_results = results
            self._state.status_message = f"Scan complete: {len(results)} results"
        except Exception as exc:
            log.exception("Scanner refresh failed")
            self._state.status_message = f"Scanner error: {exc}"

    def _refresh_factors(self) -> None:
        """Refresh factor data in background thread."""
        try:
            from quant_tools.factors.scoring import calculate_factor_scores
            from quant_tools.factors.macro import detect_regime
            from quant_tools.data.universe import get_universe

            symbols = get_universe("sp500")[:30]
            self._state.factor_scores = calculate_factor_scores(symbols)
            self._state.macro_snapshot = detect_regime()
            self._state.status_message = "Factor analysis complete"
        except Exception as exc:
            log.exception("Factor refresh failed")
            self._state.status_message = f"Factor error: {exc}"


def run_dashboard() -> None:
    """Launch the TUI dashboard."""
    from quant_tools.logging_config import setup_logging
    setup_logging()
    app = QuantToolsApp()
    app.run()
