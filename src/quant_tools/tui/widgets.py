"""TUI widget subclasses (deco-monitor pattern)."""

from __future__ import annotations

from textual.widgets import Static

from quant_tools.models import (
    BacktestResult,
    DashboardState,
    FactorScore,
    MacroSnapshot,
    PortfolioAnalysis,
    ScanResult,
)
from quant_tools.tui.formatters import (
    bar_chart,
    color_by_value,
    fmt_currency,
    fmt_pct,
    fmt_score,
)


class ScannerPanel(Static):
    """Display scanner results."""

    def update_data(self, results: list[ScanResult]) -> None:
        lines = ["[b]Scanner Results[/b]"]
        if not results:
            lines.append("  [dim]No results yet. Run a scan.[/dim]")
            self.update("\n".join(lines))
            return

        lines.append(f"  {'Symbol':<8} {'Name':<20} {'Score':>6} {'Price':>10} {'1D':>8} {'RSI':>6}")
        lines.append("  " + "-" * 62)

        for r in results[:15]:
            color = color_by_value(r.change_1d)
            rsi_str = f"{r.rsi_14:.0f}" if r.rsi_14 is not None else "N/A"
            lines.append(
                f"  {r.symbol:<8} {r.name[:20]:<20} "
                f"{fmt_score(r.score):>6} "
                f"${r.price:>9.2f} "
                f"[{color}]{r.change_1d:>+7.2f}%[/{color}] "
                f"{rsi_str:>6}"
            )

        self.update("\n".join(lines))


class BacktestPanel(Static):
    """Display backtest results."""

    def update_data(self, result: BacktestResult | None) -> None:
        if result is None:
            self.update("[b]Backtest[/b]\n  [dim]No backtest run yet.[/dim]")
            return

        ret_color = color_by_value(result.total_return)
        dd_bar = bar_chart(abs(result.max_drawdown) * 100, 50, 16)

        lines = [
            f"[b]Backtest: {result.ticker} ({result.strategy.value})[/b]",
            f"  Period: {result.start_date} to {result.end_date}"
            if result.start_date and result.end_date
            else "  Period: [dim]not recorded[/dim]",
            f"  Capital: {fmt_currency(result.initial_capital)} -> {fmt_currency(result.final_value)}",
            f"  Return: [{ret_color}]{fmt_pct(result.total_return)}[/{ret_color}]  "
            f"CAGR: {fmt_pct(result.cagr)}",
            f"  Sharpe: {result.sharpe_ratio:.2f}  Sortino: {result.sortino_ratio:.2f}",
            f"  Max DD: [red]{fmt_pct(result.max_drawdown)}[/red] [{dd_bar}] "
            f"({result.max_drawdown_duration_days}d)",
            f"  Trades: {result.total_trades}  "
            f"Win Rate: {fmt_pct(result.win_rate)}  "
            f"Profit Factor: {result.profit_factor:.2f}",
        ]
        self.update("\n".join(lines))


class PortfolioPanel(Static):
    """Display portfolio analysis."""

    def update_data(self, analysis: PortfolioAnalysis | None) -> None:
        if analysis is None:
            self.update("[b]Portfolio[/b]\n  [dim]No portfolio loaded.[/dim]")
            return

        ret_color = color_by_value(analysis.annualized_return)
        lines = [
            "[b]Portfolio Analysis[/b]",
            f"  Return: [{ret_color}]{fmt_pct(analysis.annualized_return)}[/{ret_color}] ann.  "
            f"Vol: {fmt_pct(analysis.volatility)}  "
            f"Sharpe: {analysis.sharpe_ratio:.2f}",
            f"  Max DD: [red]{fmt_pct(analysis.max_drawdown)}[/red]",
            "",
            f"  {'Symbol':<8} {'Weight':>8} {'Return':>10} {'Vol':>8} {'Sharpe':>8}",
            "  " + "-" * 46,
        ]

        for h in analysis.holdings:
            ret_c = color_by_value(h.annualized_return)
            lines.append(
                f"  {h.symbol:<8} {fmt_pct(h.weight):>8} "
                f"[{ret_c}]{fmt_pct(h.annualized_return):>10}[/{ret_c}] "
                f"{fmt_pct(h.volatility):>8} {h.sharpe:>8.2f}"
            )

        if analysis.sector_exposure:
            lines.append("")
            lines.append("  [b]Sector Exposure[/b]")
            for sector, weight in sorted(
                analysis.sector_exposure.items(), key=lambda x: -x[1]
            ):
                bar = bar_chart(weight * 100, 100, 12)
                lines.append(f"  {sector:<25} [{bar}] {fmt_pct(weight)}")

        self.update("\n".join(lines))


class FactorsPanel(Static):
    """Display factor scores and macro regime."""

    def update_data(
        self, scores: list[FactorScore], macro: MacroSnapshot | None
    ) -> None:
        lines = ["[b]Factor Analysis[/b]"]

        if macro:
            regime = macro.regime.value.upper()
            regime_colors = {
                "EXPANSION": "green",
                "RECOVERY": "cyan",
                "SLOWDOWN": "yellow",
                "CONTRACTION": "red",
            }
            color = regime_colors.get(regime, "white")
            lines.append(f"  Regime: [{color}]{regime}[/{color}]")
            if macro.gdp_growth is not None:
                lines.append(f"  GDP: {macro.gdp_growth:.1f}%  "
                             f"Unemp: {macro.unemployment or 0:.1f}%  "
                             f"CPI: {macro.cpi_yoy or 0:.1f}%")
            lines.append("")

        if not scores:
            lines.append("  [dim]No factor scores yet.[/dim]")
            self.update("\n".join(lines))
            return

        lines.append(
            f"  {'Rank':>4} {'Symbol':<8} {'Value':>7} {'Mom':>7} "
            f"{'Qual':>7} {'Comp':>7}"
        )
        lines.append("  " + "-" * 48)

        for f in scores[:15]:
            lines.append(
                f"  {f.rank:>4} {f.symbol:<8} "
                f"{f.value_z:>+7.2f} {f.momentum_z:>+7.2f} "
                f"{f.quality_z:>+7.2f} {f.composite_z:>+7.2f}"
            )

        self.update("\n".join(lines))


class StatusPanel(Static):
    """Bottom status bar."""

    def update_data(self, state: DashboardState) -> None:
        ts = state.last_update.strftime("%H:%M:%S")
        tab = state.active_tab.value.upper()
        loading = " [yellow]Loading...[/yellow]" if state.is_loading else ""
        self.update(
            f"Tab: {tab} | Updated: {ts} | {state.status_message}{loading}"
        )
