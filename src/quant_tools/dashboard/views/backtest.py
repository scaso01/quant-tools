"""Strategy Backtester view — run and compare trading strategies."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quant_tools.backtest.engine import run_backtest
from quant_tools.backtest.strategies import (
    bollinger_strategy,
    momentum_strategy,
    rsi_strategy,
    sma_crossover,
)
from quant_tools.dashboard.core.colors import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    TEXT_MUTED,
)
from quant_tools.dashboard.core.date_presets import render_date_presets
from quant_tools.dashboard.core.helpers import (
    PLOTLY_LAYOUT,
    directional_color,
    format_currency,
    format_number,
    format_pct,
    render_metric_card,
    render_news_section,
)
from quant_tools.dashboard.core.page_help import render_page_help
from quant_tools.dashboard.core.symbol_picker import render_symbol_picker
from quant_tools.dashboard.core.symbol_state import get_active_symbol, set_active_symbol
from quant_tools.data.market_data import fetch_prices
from quant_tools.errors import ComputeError, DataError, NetworkError
from quant_tools.models import BacktestResult, StrategyType

# ── Strategy registry ────────────────────────────────────────────────

STRATEGY_MAP: dict[str, StrategyType] = {
    "SMA Crossover": StrategyType.SMA_CROSSOVER,
    "Bollinger Bands": StrategyType.BOLLINGER,
    "RSI": StrategyType.RSI,
    "Momentum": StrategyType.MOMENTUM,
}

STRATEGY_COLORS: dict[StrategyType, str] = {
    StrategyType.SMA_CROSSOVER: ACCENT_BLUE,
    StrategyType.BOLLINGER: ACCENT_PURPLE,
    StrategyType.RSI: ACCENT_AMBER,
    StrategyType.MOMENTUM: ACCENT_ORANGE,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _generate_positions(
    strategy_name: str,
    close: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Dispatch to the correct strategy function and return positions."""
    if strategy_name == "SMA Crossover":
        return sma_crossover(close, fast=params["fast"], slow=params["slow"])
    if strategy_name == "Bollinger Bands":
        return bollinger_strategy(
            close, period=params["period"], num_std=params["num_std"]
        )
    if strategy_name == "RSI":
        return rsi_strategy(
            close,
            period=params["period"],
            oversold=params["oversold"],
            overbought=params["overbought"],
        )
    if strategy_name == "Momentum":
        return momentum_strategy(close, lookback=params["lookback"])
    msg = f"Unknown strategy: {strategy_name}"
    raise ValueError(msg)


def _default_params(strategy_name: str) -> dict:
    """Return default parameter dict for a strategy (used in comparison)."""
    if strategy_name == "SMA Crossover":
        return {"fast": 20, "slow": 50}
    if strategy_name == "Bollinger Bands":
        return {"period": 20, "num_std": 2.0}
    if strategy_name == "RSI":
        return {"period": 14, "oversold": 30.0, "overbought": 70.0}
    if strategy_name == "Momentum":
        return {"lookback": 60}
    return {}


def _compute_drawdown_series(equity_curve: list[float]) -> list[float]:
    """Compute drawdown as fraction from running peak for each point."""
    eq = np.array(equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(eq)
    drawdown = np.where(running_max > 0, (running_max - eq) / running_max, 0.0)
    return drawdown.tolist()


# ── Main render ──────────────────────────────────────────────────────


def render() -> None:
    """Render the Strategy Backtester page."""

    # ── Header ────────────────────────────────────────────────────
    st.markdown(
        '<div class="dashboard-header">'
        '<h1 class="dashboard-title">Strategy Backtester</h1>'
        '<p class="dashboard-subtitle">'
        "Backtest trading strategies on historical data"
        "</p></div>",
        unsafe_allow_html=True,
    )

    render_page_help("backtest")

    # ── Sidebar inputs ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Backtest Settings")

        _default_sym = get_active_symbol() or "AAPL"
        ticker = render_symbol_picker(
            label="Ticker",
            key="bt_symbol_picker",
            default=_default_sym,
            sidebar=False,  # already inside st.sidebar context
        ) or _default_sym

        strategy_name = st.selectbox(
            "Strategy",
            list(STRATEGY_MAP.keys()),
            key="bt_strategy",
            help="The trading strategy to test against historical data",
        )

        start_date, end_date = render_date_presets(
            "bt", default="1Y", sidebar=False,
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            key="bt_capital",
            help="Starting investment amount in dollars",
        )

        commission = st.slider(
            "Commission",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
            key="bt_commission",
            help="Trading fee per transaction as a decimal (0.001 = 0.1%)",
        )

        # ── Strategy-specific parameters ──────────────────────────
        with st.expander("Strategy Parameters"):
            if strategy_name == "SMA Crossover":
                fast = st.slider(
                    "Fast Period", 5, 50, 20, key="bt_sma_fast",
                    help="Short moving average window (days) — triggers buy signals",
                )
                slow = st.slider(
                    "Slow Period", 20, 200, 50, key="bt_sma_slow",
                    help="Long moving average window (days) — buy when fast crosses above",
                )
                params = {"fast": fast, "slow": slow}

            elif strategy_name == "Bollinger Bands":
                bb_period = st.slider(
                    "Period", 10, 50, 20, key="bt_bb_period",
                    help="Rolling window for the Bollinger Band center line",
                )
                bb_std = st.slider(
                    "Std Deviations",
                    1.0,
                    3.0,
                    2.0,
                    step=0.1,
                    key="bt_bb_std",
                    help="Band width — higher values mean wider bands, fewer trades",
                )
                params = {"period": bb_period, "num_std": bb_std}

            elif strategy_name == "RSI":
                rsi_period = st.slider(
                    "Period", 5, 30, 14, key="bt_rsi_period",
                    help="RSI calculation window — 14 is the standard default",
                )
                oversold = st.slider(
                    "Oversold", 10, 40, 30, key="bt_rsi_oversold",
                    help="Buy when RSI drops below this level (stock is oversold)",
                )
                overbought = st.slider(
                    "Overbought", 60, 90, 70, key="bt_rsi_overbought",
                    help="Sell when RSI rises above this level (stock is overbought)",
                )
                params = {
                    "period": rsi_period,
                    "oversold": float(oversold),
                    "overbought": float(overbought),
                }

            elif strategy_name == "Momentum":
                lookback = st.slider(
                    "Lookback", 20, 120, 60, key="bt_mom_lookback",
                    help="Days to look back when measuring price momentum",
                )
                params = {"lookback": lookback}

            else:
                params = {}

        run_clicked = st.button(
            "Run Backtest",
            width="stretch",
            type="primary",
            key="bt_run",
        )

    # ── Run backtest ──────────────────────────────────────────────
    if not run_clicked:
        st.info(
            "Configure your backtest in the sidebar, or try one of the "
            "quick-start examples below."
        )

        st.markdown("#### Quick Start")
        st.markdown("Try one of these example backtests:")
        _examples = [
            ("AAPL", "SMA Crossover", "Apple with moving average crossover"),
            ("MSFT", "RSI", "Microsoft with RSI mean reversion"),
            ("NVDA", "Momentum", "NVIDIA with momentum strategy"),
        ]
        _ex_cols = st.columns(len(_examples))
        for col, (ex_ticker, ex_strategy, ex_desc) in zip(_ex_cols, _examples):
            with col:
                st.markdown(f"**{ex_ticker} + {ex_strategy}**")
                st.caption(ex_desc)
                if st.button(
                    "Run",
                    key=f"example_{ex_ticker}_{ex_strategy}",
                    width="stretch",
                    type="primary",
                ):
                    set_active_symbol(ex_ticker)
                    st.session_state["bt_strategy"] = ex_strategy
                    st.rerun()
        return

    strategy_type = STRATEGY_MAP[strategy_name]

    try:
        with st.spinner("Running backtest..."):
            df = fetch_prices(
                ticker, start=str(start_date), end=str(end_date)
            )
            close = df["Close"].values

            positions = _generate_positions(strategy_name, close, params)

            result: BacktestResult = run_backtest(
                close,
                positions,
                initial_capital=float(initial_capital),
                commission=commission,
                strategy=strategy_type,
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                params=params,
            )
    except DataError as exc:
        st.warning(f"Data issue: {exc}")
        return
    except NetworkError as exc:
        st.error(f"Network error: {exc}")
        return
    except ComputeError as exc:
        st.error(f"Computation error: {exc}")
        return

    # ── KPI row ───────────────────────────────────────────────────
    kpi_cols = st.columns(6)

    with kpi_cols[0]:
        render_metric_card(
            "Total Return",
            format_pct(result.total_return),
            directional_color(result.total_return),
            glossary_key="CAGR",
            delta=result.total_return * 100,
            delta_suffix="%",
            size="large",
        )
    with kpi_cols[1]:
        render_metric_card(
            "CAGR",
            format_pct(result.cagr),
            directional_color(result.cagr),
            glossary_key="CAGR",
            delta=result.cagr * 100,
            delta_suffix="%",
        )
    with kpi_cols[2]:
        render_metric_card(
            "Sharpe",
            format_number(result.sharpe_ratio),
            directional_color(result.sharpe_ratio - 1.0),
            glossary_key="Sharpe Ratio",
        )
    with kpi_cols[3]:
        render_metric_card(
            "Sortino",
            format_number(result.sortino_ratio),
            directional_color(result.sortino_ratio - 1.0),
            glossary_key="Sortino Ratio",
        )
    with kpi_cols[4]:
        render_metric_card(
            "Max Drawdown",
            format_pct(result.max_drawdown),
            ACCENT_RED,
            glossary_key="Max Drawdown",
            delta=-abs(result.max_drawdown) * 100,
            delta_suffix="%",
        )
    with kpi_cols[5]:
        render_metric_card(
            "Win Rate",
            format_pct(result.win_rate),
            directional_color(result.win_rate - 0.5),
            glossary_key="Win Rate",
        )

    # ── Extra stats row ───────────────────────────────────────────
    stat_cols = st.columns(4)
    with stat_cols[0]:
        render_metric_card(
            "Final Value",
            format_currency(result.final_value),
            ACCENT_BLUE,
        )
    with stat_cols[1]:
        render_metric_card(
            "Total Trades",
            format_number(result.total_trades, 0),
            ACCENT_BLUE,
        )
    with stat_cols[2]:
        render_metric_card(
            "Profit Factor",
            format_number(result.profit_factor),
            directional_color(result.profit_factor - 1.0),
            glossary_key="Profit Factor",
        )
    with stat_cols[3]:
        render_metric_card(
            "Max DD Duration",
            f"{result.max_drawdown_duration_days}d",
            ACCENT_RED,
            glossary_key="Drawdown",
        )

    # ── Next Steps ────────────────────────────────────────────────
    st.markdown("---")
    next_cols = st.columns([1, 1, 4])
    with next_cols[0]:
        if st.button("Build a Portfolio \u2192", key="backtest_to_portfolio"):
            st.session_state["active_page"] = "\U0001f4bc Portfolio Analyzer"
            st.query_params["page"] = "\U0001f4bc Portfolio Analyzer"
            st.rerun()
    with next_cols[1]:
        if st.button("Back to Scanner \u2192", key="backtest_to_scanner"):
            st.session_state["active_page"] = "\U0001f4e1 Market Scanner"
            st.query_params["page"] = "\U0001f4e1 Market Scanner"
            st.rerun()

    st.divider()

    # ── Equity curve + drawdown ───────────────────────────────────
    st.markdown("### Equity Curve")

    equity = result.equity_curve
    drawdown = _compute_drawdown_series(equity)

    # Build x-axis from date range
    n_points = len(equity)
    if n_points > 1:
        delta = (end_date - start_date) / (n_points - 1)
        x_dates = [start_date + delta * i for i in range(n_points)]
    else:
        x_dates = [start_date]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=equity,
            name="Equity",
            fill="tozeroy",
            line={"color": ACCENT_GREEN, "width": 2},
            fillcolor="rgba(16, 185, 129, 0.15)",
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=[-d * 100 for d in drawdown],
            name="Drawdown %",
            line={"color": ACCENT_RED, "width": 1, "dash": "dot"},
            yaxis="y2",
        )
    )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=450,
        yaxis={"title": "Portfolio Value ($)", "side": "left"},
        yaxis2={
            "title": "Drawdown (%)",
            "side": "right",
            "overlaying": "y",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": -0.15},
    )

    st.plotly_chart(fig, width="stretch")

    # ── Strategy comparison ───────────────────────────────────────
    with st.expander("Compare All Strategies"):
        try:
            with st.spinner("Running all strategies..."):
                comparison_results: list[BacktestResult] = []

                for name, stype in STRATEGY_MAP.items():
                    default_p = _default_params(name)
                    pos = _generate_positions(name, close, default_p)
                    res = run_backtest(
                        close,
                        pos,
                        initial_capital=float(initial_capital),
                        commission=commission,
                        strategy=stype,
                        ticker=ticker.upper(),
                        start_date=start_date,
                        end_date=end_date,
                        params=default_p,
                    )
                    comparison_results.append(res)

            strategy_names = list(STRATEGY_MAP.keys())
            returns = [r.total_return * 100 for r in comparison_results]
            sharpes = [r.sharpe_ratio for r in comparison_results]
            colors = [
                STRATEGY_COLORS.get(r.strategy, ACCENT_BLUE)
                for r in comparison_results
            ]

            comp_fig = go.Figure()

            comp_fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=returns,
                    name="Total Return (%)",
                    marker_color=colors,
                    yaxis="y",
                )
            )

            comp_fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=sharpes,
                    name="Sharpe Ratio",
                    marker_color=[TEXT_MUTED] * len(sharpes),
                    yaxis="y2",
                )
            )

            comp_fig.update_layout(**PLOTLY_LAYOUT)
            comp_fig.update_layout(
                barmode="group",
                height=400,
                title="Strategy Comparison (Default Parameters)",
                yaxis={"title": "Total Return (%)"},
                yaxis2={
                    "title": "Sharpe Ratio",
                    "side": "right",
                    "overlaying": "y",
                    "showgrid": False,
                },
                legend={"orientation": "h", "y": -0.2},
            )

            st.plotly_chart(comp_fig, width="stretch")

            # Summary table
            summary_data = {
                "Strategy": strategy_names,
                "Return": [format_pct(r.total_return) for r in comparison_results],
                "CAGR": [format_pct(r.cagr) for r in comparison_results],
                "Sharpe": [format_number(r.sharpe_ratio) for r in comparison_results],
                "Max DD": [
                    format_pct(r.max_drawdown) for r in comparison_results
                ],
                "Trades": [r.total_trades for r in comparison_results],
            }
            st.dataframe(summary_data, width="stretch")

        except (DataError, NetworkError, ComputeError) as exc:
            st.warning(f"Could not run comparison: {exc}")

    # ── News ──────────────────────────────────────────────────────
    st.divider()
    render_news_section([ticker.upper()])
