"""Rolling Analytics page — interactive rolling-window metrics.

Shows rolling Sharpe, volatility, beta, and correlation with configurable
window sizes via sidebar controls.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_rolling_sharpe(
    returns: pd.Series, window: int = 63, periods_per_year: int = 252
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    return sharpe.dropna()


def _compute_rolling_volatility(
    returns: pd.Series, window: int = 63, periods_per_year: int = 252
) -> pd.Series:
    """Rolling annualized volatility."""
    return (returns.rolling(window).std() * np.sqrt(periods_per_year)).dropna()


def _compute_rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta vs benchmark."""
    aligned = pd.concat(
        [returns.rename("r"), benchmark_returns.rename("b")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    cov = aligned["r"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    beta = (cov / var).dropna()
    return beta


def _line_chart(series: pd.Series, title: str, color: str, y_title: str):
    """Single-series line chart using the dashboard's shared plotly styling."""
    import plotly.graph_objects as go

    from quant_tools.dashboard.core.helpers import PLOTLY_LAYOUT

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": color, "width": 2},
            name=title,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, title=title, xaxis_title="Date", yaxis_title=y_title
    )
    return fig


def render() -> None:
    """Render the Rolling Analytics dashboard page.

    Uses streamlit (imported at call time to avoid import-time side effects).
    """
    import streamlit as st

    from quant_tools.dashboard.core.colors import (
        ACCENT_AMBER,
        ACCENT_BLUE,
        ACCENT_GREEN,
    )
    from quant_tools.dashboard.core.date_presets import render_date_presets
    from quant_tools.dashboard.core.helpers import (
        directional_color,
        format_number,
        format_pct,
        render_metric_card,
    )
    from quant_tools.dashboard.core.page_help import render_page_help
    from quant_tools.dashboard.core.symbol_picker import render_symbol_picker
    from quant_tools.dashboard.core.symbol_state import get_active_symbol
    from quant_tools.data.market_data import fetch_prices
    from quant_tools.errors import ComputeError, DataError, NetworkError
    from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS

    st.header("Rolling Analytics")
    st.markdown("Analyze rolling-window risk and return metrics over time.")

    render_page_help("rolling")

    # \u2500\u2500 Sidebar controls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with st.sidebar:
        st.markdown("### Rolling Settings")

        symbol = render_symbol_picker(
            label="Ticker",
            key="roll_symbol_picker",
            default=get_active_symbol() or "AAPL",
            sidebar=False,  # already inside st.sidebar
        ) or "AAPL"

        # Only single-symbol benchmarks work here; a rolling beta needs one
        # price series, and composites like 60/40 carry weights instead.
        benchmark_keys = [
            k for k, v in STANDARD_BENCHMARKS.items() if v.symbol
        ]
        benchmark = st.selectbox(
            "Benchmark",
            benchmark_keys,
            index=benchmark_keys.index("SPY") if "SPY" in benchmark_keys else 0,
            key="roll_benchmark",
            help="The series to measure rolling beta against",
        )

        window = st.slider(
            "Window (days)",
            min_value=10,
            max_value=252,
            value=63,
            step=1,
            key="roll_window",
            help="How many trading days each rolling metric looks back over",
        )

        start_date, end_date = render_date_presets(
            "roll", default="2Y", sidebar=False,
        )

        run_analysis = st.button(
            "Run Analysis", type="primary", key="roll_run", width="stretch"
        )

    # Rendered before the fetch so the page always states what it is analyzing,
    # including on the short-history and data-error paths below.
    st.markdown(
        f"#### {symbol} vs {benchmark} \u00b7 {window}-day window "
        f"\u00b7 {start_date} to {end_date}"
    )

    # Nothing is fetched until asked, matching Scanner's "Run Scan" and
    # Backtest's "Run Backtest". Navigating to a page should never block on
    # the network.
    if run_analysis:
        st.session_state["roll_has_run"] = True
    if not st.session_state.get("roll_has_run"):
        st.info(
            "Choose a ticker, benchmark and window, then press "
            "**Run Analysis** to compute rolling metrics."
        )
        return

    # \u2500\u2500 Load prices \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    try:
        with st.spinner(f"Loading {symbol}..."):
            prices = fetch_prices(symbol, start=str(start_date), end=str(end_date))
            bench_prices = fetch_prices(
                benchmark, start=str(start_date), end=str(end_date)
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

    returns = prices["Close"].pct_change().dropna()
    bench_returns = bench_prices["Close"].pct_change().dropna()

    if len(returns) <= window:
        st.warning(
            f"Not enough history for a {window}-day window: {len(returns)} "
            "trading days loaded. Choose a longer date range or a shorter window."
        )
        return

    sharpe = _compute_rolling_sharpe(returns, window=window)
    volatility = _compute_rolling_volatility(returns, window=window)
    beta = _compute_rolling_beta(returns, bench_returns, window=window)

    # \u2500\u2500 Latest values \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    cols = st.columns(4)
    latest_sharpe = float(sharpe.iloc[-1]) if not sharpe.empty else 0.0
    latest_vol = float(volatility.iloc[-1]) if not volatility.empty else 0.0
    latest_beta = float(beta.iloc[-1]) if not beta.empty else 0.0

    with cols[0]:
        render_metric_card(
            "Rolling Sharpe",
            format_number(latest_sharpe),
            directional_color(latest_sharpe),
            glossary_key="Sharpe Ratio",
        )
    with cols[1]:
        render_metric_card(
            "Rolling Volatility", format_pct(latest_vol), ACCENT_AMBER
        )
    with cols[2]:
        render_metric_card(
            f"Rolling Beta vs {benchmark}",
            format_number(latest_beta),
            directional_color(1.0 - latest_beta),
            glossary_key="Beta",
        )
    with cols[3]:
        render_metric_card("Window", f"{window} days", ACCENT_BLUE)

    st.divider()

    # \u2500\u2500 Charts \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if not sharpe.empty:
        st.plotly_chart(
            _line_chart(
                sharpe, f"{symbol} Rolling Sharpe Ratio", ACCENT_GREEN, "Sharpe"
            ),
            width="stretch",
        )
    if not volatility.empty:
        st.plotly_chart(
            _line_chart(
                volatility,
                f"{symbol} Rolling Volatility (annualized)",
                ACCENT_AMBER,
                "Volatility",
            ),
            width="stretch",
        )
    if not beta.empty:
        st.plotly_chart(
            _line_chart(
                beta, f"{symbol} Rolling Beta vs {benchmark}", ACCENT_BLUE, "Beta"
            ),
            width="stretch",
        )
    else:
        st.info(
            f"Not enough overlapping history with {benchmark} to compute a "
            "rolling beta over this window."
        )
