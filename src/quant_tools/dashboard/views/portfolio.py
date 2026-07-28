"""Portfolio Analyzer view — holdings analysis, correlation, and Monte Carlo."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_tools.dashboard.core.colors import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
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
from quant_tools.data.market_data import fetch_prices
from quant_tools.errors import ComputeError, DataError, NetworkError
from quant_tools.models import (
    AssetConstraint,
    BenchmarkDefinition,
    DistributionType,
    GroupConstraint,
    MonteCarloResult,
    OptimizationConstraints,
    PortfolioAnalysis,
)
from quant_tools.portfolio.analyzer import analyze_portfolio
from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS, BenchmarkAnalyzer
from quant_tools.portfolio.cross_correlation import (
    calculate_cross_asset_correlations,
    calculate_rolling_correlations,
)
from quant_tools.portfolio.monte_carlo import run_monte_carlo

log = logging.getLogger("quant_tools")

# ── Preset Portfolios ────────────────────────────────────────────────

PRESETS: dict[str, dict[str, float]] = {
    "60/40": {"SPY": 0.6, "BND": 0.4},
    "Tech Heavy": {
        "AAPL": 0.25,
        "MSFT": 0.25,
        "NVDA": 0.20,
        "GOOGL": 0.15,
        "BND": 0.15,
    },
    "All-Weather": {
        "SPY": 0.30,
        "TLT": 0.40,
        "GC=F": 0.15,
        "DBA": 0.075,
        "BND": 0.075,
    },
}

DEFAULT_HOLDINGS = '{"AAPL": 0.3, "MSFT": 0.2, "BND": 0.3, "GLD": 0.2}'


# ── Render ────────────────────────────────────────────────────────────


def render() -> None:
    """Main render function called by the dashboard app."""

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>Portfolio Analyzer</h1>"
        f'<p style="color:{TEXT_MUTED};">Analyze holdings, correlations, and '
        "Monte Carlo projections</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    render_page_help("portfolio")

    # ── Sidebar Inputs ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Portfolio Settings")

        # Preset buttons
        st.markdown("**Presets**")
        preset_cols = st.columns(len(PRESETS))
        for col, (name, weights) in zip(preset_cols, PRESETS.items()):
            with col:
                if st.button(name, key=f"preset_{name}", use_container_width=True):
                    st.session_state["portfolio_holdings_list"] = [
                        (sym, wt) for sym, wt in weights.items()
                    ]
                    st.rerun()

        # ── Visual Holdings Builder ──────────────────────────────
        st.markdown("**Holdings**")

        # Initialize holdings list from JSON fallback or defaults
        if "portfolio_holdings_list" not in st.session_state:
            try:
                _init_json = st.session_state.get(
                    "portfolio_holdings_json", DEFAULT_HOLDINGS
                )
                _init = json.loads(_init_json) if isinstance(_init_json, str) else {}
                st.session_state["portfolio_holdings_list"] = [
                    (sym, wt) for sym, wt in _init.items()
                ]
            except (ValueError, TypeError):
                st.session_state["portfolio_holdings_list"] = [
                    ("AAPL", 0.3), ("MSFT", 0.2), ("BND", 0.3), ("GLD", 0.2),
                ]

        _holdings_list: list[tuple[str, float]] = st.session_state[
            "portfolio_holdings_list"
        ]
        _to_remove: int | None = None

        for idx, (sym, wt) in enumerate(_holdings_list):
            c_sym, c_wt, c_del = st.columns([3, 2, 1])
            with c_sym:
                new_sym = st.text_input(
                    "Ticker",
                    value=sym,
                    key=f"hold_sym_{idx}",
                    label_visibility="collapsed",
                    placeholder="AAPL",
                )
            with c_wt:
                new_wt = st.number_input(
                    "Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(wt),
                    step=0.05,
                    format="%.2f",
                    key=f"hold_wt_{idx}",
                    label_visibility="collapsed",
                )
            with c_del:
                if st.button("\u2715", key=f"hold_del_{idx}"):
                    _to_remove = idx

            # Update in-place if changed
            if (new_sym, new_wt) != (sym, wt):
                _holdings_list[idx] = (new_sym.upper().strip(), new_wt)
                st.session_state["portfolio_holdings_list"] = _holdings_list

        # Remove holding
        if _to_remove is not None:
            _holdings_list.pop(_to_remove)
            st.session_state["portfolio_holdings_list"] = _holdings_list
            st.rerun()

        # Add holding button
        if st.button("+ Add Holding", key="hold_add", use_container_width=True):
            _holdings_list.append(("", 0.10))
            st.session_state["portfolio_holdings_list"] = _holdings_list
            st.rerun()

        # Weight sum indicator
        _weight_sum = sum(wt for _, wt in _holdings_list)
        _sum_color = ACCENT_GREEN if abs(_weight_sum - 1.0) < 0.01 else ACCENT_AMBER
        st.markdown(
            f'<div style="text-align:right;font-size:0.8rem;color:{_sum_color};">'
            f"Sum: {_weight_sum:.0%}</div>",
            unsafe_allow_html=True,
        )
        if abs(_weight_sum - 1.0) >= 0.01 and _weight_sum > 0:
            if st.button("Normalize to 100%", key="hold_norm", use_container_width=True):
                _holdings_list = [
                    (sym, wt / _weight_sum) for sym, wt in _holdings_list
                ]
                st.session_state["portfolio_holdings_list"] = _holdings_list
                st.rerun()

        # Convert list to JSON for backward compat
        holdings_json = json.dumps(
            {sym: wt for sym, wt in _holdings_list if sym.strip()},
            indent=2,
        )
        st.session_state["portfolio_holdings_json"] = holdings_json

        # JSON fallback for power users
        with st.expander("Advanced: JSON Mode"):
            _json_edit = st.text_area(
                "Holdings JSON",
                value=holdings_json,
                height=80,
                key="portfolio_json_fallback",
                help="Edit directly as JSON. Changes here override the visual builder.",
            )
            if _json_edit != holdings_json:
                try:
                    _parsed = json.loads(_json_edit)
                    if isinstance(_parsed, dict):
                        st.session_state["portfolio_holdings_list"] = [
                            (sym, wt) for sym, wt in _parsed.items()
                        ]
                        st.session_state["portfolio_holdings_json"] = _json_edit
                        st.rerun()
                except (ValueError, TypeError):
                    st.error("Invalid JSON")

        period = st.selectbox(
            "Period",
            options=["3mo", "6mo", "1y", "2y"],
            index=2,
            key="portfolio_period",
            help="How far back to analyze — longer periods smooth out noise",
        )

        risk_free_rate = st.slider(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=0.10,
            value=0.04,
            step=0.005,
            format="%.3f",
            key="portfolio_rfr",
            help="Baseline return for risk-free assets (e.g., Treasury bills)",
        )

        # ── Benchmark selector ─────────────────────────────────────
        st.markdown("### Benchmark")
        benchmark_options = {k: v.name for k, v in STANDARD_BENCHMARKS.items()}
        benchmark_options["custom"] = "Custom Upload"
        benchmark_options["none"] = "None"
        benchmark_choice = st.selectbox(
            "Benchmark",
            options=list(benchmark_options.keys()),
            format_func=lambda k: benchmark_options[k],
            index=len(benchmark_options) - 1,  # default "None"
            key="portfolio_benchmark",
            help="Index to compare your portfolio against (e.g., S&P 500)",
        )

        custom_benchmark_file = None
        if benchmark_choice == "custom":
            custom_benchmark_file = st.file_uploader(
                "Upload CSV (date, return columns)",
                type=["csv"],
                key="portfolio_benchmark_csv",
                help="CSV with 'date' and 'return' columns. Returns as decimals.",
            )

        st.markdown("### Monte Carlo")

        n_sims = st.slider(
            "Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="portfolio_mc_sims",
            help="Number of random scenarios to simulate (more = smoother results)",
        )

        horizon = st.slider(
            "Trading Days",
            min_value=63,
            max_value=504,
            value=252,
            step=1,
            key="portfolio_mc_horizon",
            help="How many trading days to project forward (252 = ~1 year)",
        )

        # ── Distribution selector ──────────────────────────────────
        dist_options = {
            "Normal": DistributionType.NORMAL,
            "Student-t": DistributionType.STUDENT_T,
            "Skew-Normal": DistributionType.SKEW_NORMAL,
            "Johnson SU": DistributionType.JOHNSON_SU,
            "GED": DistributionType.GED,
            "Auto-detect (BIC)": None,
        }
        dist_choice = st.selectbox(
            "Distribution",
            list(dist_options.keys()),
            key="portfolio_mc_dist",
            help="Statistical model for returns — Auto-detect picks the best fit",
        )

        run_analysis = st.button(
            "Analyze Portfolio",
            type="primary",
            use_container_width=True,
            key="portfolio_run",
        )

    # ── Parse Holdings ────────────────────────────────────────────────
    if not run_analysis:
        st.markdown(
            "#### Quick Start\n"
            "Select a preset portfolio below to get started instantly, "
            "or enter custom holdings as JSON in the sidebar."
        )

        _PRESET_DESCRIPTIONS: dict[str, str] = {
            "60/40": "Classic balanced allocation -- 60% stocks (SPY), 40% bonds (BND). "
                     "A time-tested mix of growth and stability.",
            "Tech Heavy": "Concentrated technology bet -- AAPL, MSFT, NVDA, GOOGL with a "
                          "small bond cushion. High growth potential, higher volatility.",
            "All-Weather": "Ray Dalio-inspired diversification -- stocks, long-term treasuries, "
                           "gold, and commodities. Designed to perform across economic regimes.",
        }

        preset_cols = st.columns(len(PRESETS))
        for col, (name, weights) in zip(preset_cols, PRESETS.items()):
            with col:
                st.markdown(f"**{name}**")
                st.caption(_PRESET_DESCRIPTIONS.get(name, ""))
                tickers = ", ".join(weights.keys())
                st.markdown(
                    f'<span style="color:{TEXT_MUTED};font-size:0.8rem;">'
                    f"{tickers}</span>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"Load {name}",
                    key=f"empty_preset_{name}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state["portfolio_holdings_json"] = json.dumps(
                        weights, indent=2,
                    )
                    st.rerun()

        st.markdown("---")
        st.markdown(
            f'<div style="color:{TEXT_MUTED};font-size:0.85rem;">'
            "Or enter custom holdings in the sidebar as JSON, e.g. "
            f"<code>{DEFAULT_HOLDINGS}</code></div>",
            unsafe_allow_html=True,
        )
        return

    try:
        holdings: dict[str, float] = json.loads(holdings_json)
    except (ValueError, TypeError):
        st.error("Invalid JSON. Please enter a valid JSON object mapping tickers to weights.")
        return

    if not holdings or not isinstance(holdings, dict):
        st.error("Holdings must be a non-empty JSON object, e.g. {\"AAPL\": 0.5, \"BND\": 0.5}")
        return

    # ── Run Portfolio Analysis ────────────────────────────────────────
    analysis: PortfolioAnalysis | None = None
    with st.spinner("Analyzing portfolio..."):
        try:
            analysis = analyze_portfolio(
                holdings=holdings,
                period=period,
                risk_free_rate=risk_free_rate,
            )
        except DataError as exc:
            st.error(f"Data error: {exc}")
            return
        except NetworkError as exc:
            st.error(f"Network error: {exc}")
            return
        except ComputeError as exc:
            st.error(f"Computation error: {exc}")
            return

    if analysis is None:
        st.error("Analysis returned no result.")
        return

    # ── KPI Row ───────────────────────────────────────────────────────
    kpi_cols = st.columns(5)

    with kpi_cols[0]:
        render_metric_card(
            "Total Return",
            format_pct(analysis.total_return),
            directional_color(analysis.total_return),
            glossary_key="CAGR",
            delta=analysis.total_return,
            delta_suffix="%",
            size="large",
        )
    with kpi_cols[1]:
        render_metric_card(
            "Ann. Return",
            format_pct(analysis.annualized_return),
            directional_color(analysis.annualized_return),
            delta=analysis.annualized_return,
            delta_suffix="%",
        )
    with kpi_cols[2]:
        render_metric_card(
            "Volatility",
            format_pct(analysis.volatility),
            ACCENT_AMBER,
            glossary_key="Annualized Volatility",
        )
    with kpi_cols[3]:
        render_metric_card(
            "Sharpe Ratio",
            format_number(analysis.sharpe_ratio),
            directional_color(analysis.sharpe_ratio),
            glossary_key="Sharpe Ratio",
            delta=analysis.sharpe_ratio,
        )
    with kpi_cols[4]:
        render_metric_card(
            "Max Drawdown",
            format_pct(analysis.max_drawdown),
            ACCENT_RED,
            glossary_key="Max Drawdown",
        )

    # ── Next Steps ────────────────────────────────────────────────────
    st.markdown("---")
    next_cols = st.columns([1, 1, 4])
    with next_cols[0]:
        if st.button("Check Factor Exposure \u2192", key="portfolio_to_factors"):
            st.session_state["active_page"] = "\U0001f3db\ufe0f Factors & Macro"
            st.query_params["page"] = "\U0001f3db\ufe0f Factors & Macro"
            st.rerun()
    with next_cols[1]:
        if st.button("View Risk Deep-Dive \u2192", key="portfolio_to_risk"):
            st.session_state["active_page"] = "\U0001f6e1\ufe0f Risk Dashboard"
            st.query_params["page"] = "\U0001f6e1\ufe0f Risk Dashboard"
            st.rerun()

    # ── Benchmark Metrics (progressive disclosure) ─────────────────────
    if benchmark_choice != "none":
        with st.expander("Benchmark Comparison", expanded=False):
            with st.spinner("Computing benchmark metrics..."):
                try:
                    bm_analyzer = BenchmarkAnalyzer()

                    # Build benchmark definition
                    if benchmark_choice == "custom" and custom_benchmark_file is not None:
                        csv_df = pd.read_csv(custom_benchmark_file)
                        bm_returns_array = csv_df["return"].values.astype(np.float64)
                        bm_def = BenchmarkDefinition(
                            name="Custom Upload",
                            returns=bm_returns_array,
                        )
                        bm_name = "Custom Upload"
                    elif benchmark_choice in STANDARD_BENCHMARKS:
                        bm_def = STANDARD_BENCHMARKS[benchmark_choice]
                        bm_name = bm_def.name
                    else:
                        bm_def = None
                        bm_name = ""

                    if bm_def is not None:
                        bm_returns = bm_analyzer.fetch_benchmark_returns(bm_def, period=period)

                        # Build portfolio returns for comparison
                        all_port_ret: list[np.ndarray] = []
                        for h in analysis.holdings:
                            df_bm = fetch_prices(h.symbol, period=period)
                            ret_bm = df_bm["Close"].pct_change().dropna().values
                            all_port_ret.append(ret_bm * h.weight)
                        min_len_bm = min(len(r) for r in all_port_ret)
                        port_ret_arr = np.sum(
                            [r[:min_len_bm] for r in all_port_ret], axis=0
                        )

                        bm_metrics = bm_analyzer.calculate_benchmark_metrics(
                            portfolio_returns=port_ret_arr,
                            benchmark_returns=bm_returns,
                            benchmark_name=bm_name,
                            risk_free_rate=risk_free_rate,
                        )

                        bm_cols = st.columns(5)

                        with bm_cols[0]:
                            render_metric_card(
                                "Alpha",
                                format_pct(bm_metrics.alpha),
                                directional_color(bm_metrics.alpha),
                                glossary_key="Alpha",
                                delta=bm_metrics.alpha,
                                delta_suffix="%",
                            )
                        with bm_cols[1]:
                            render_metric_card(
                                "Beta",
                                format_number(bm_metrics.beta),
                                ACCENT_BLUE,
                                glossary_key="Beta",
                            )
                        with bm_cols[2]:
                            render_metric_card(
                                "Info Ratio",
                                format_number(bm_metrics.information_ratio),
                                directional_color(bm_metrics.information_ratio),
                                glossary_key="Information Ratio",
                                delta=bm_metrics.information_ratio,
                            )
                        with bm_cols[3]:
                            render_metric_card(
                                "Treynor",
                                format_number(bm_metrics.treynor_ratio),
                                directional_color(bm_metrics.treynor_ratio),
                                glossary_key="Treynor Ratio",
                                delta=bm_metrics.treynor_ratio,
                            )
                        with bm_cols[4]:
                            render_metric_card(
                                "Tracking Err",
                                format_pct(bm_metrics.tracking_error),
                                ACCENT_ORANGE,
                                glossary_key="Tracking Error",
                            )
                    else:
                        if benchmark_choice == "custom":
                            st.info("Upload a CSV file to compare against a custom benchmark.")
                except (DataError, NetworkError) as exc:
                    st.warning(f"Could not compute benchmark metrics: {exc}")
                except Exception as exc:
                    st.warning(f"Benchmark error: {exc}")

    # ── Holdings Table ────────────────────────────────────────────────
    st.markdown("### Holdings")
    holdings_rows = []
    for h in analysis.holdings:
        holdings_rows.append(
            {
                "Symbol": h.symbol,
                "Weight": f"{h.weight:.1%}",
                "Price": f"${h.current_price:,.2f}",
                "Total Return": format_pct(h.total_return),
                "Ann. Return": format_pct(h.annualized_return),
                "Volatility": format_pct(h.volatility),
                "Sharpe": format_number(h.sharpe),
            }
        )
    st.dataframe(holdings_rows, use_container_width=True, hide_index=True)

    # ── Charts Row ────────────────────────────────────────────────────
    chart_left, chart_right = st.columns(2)

    # Sector Exposure Donut
    with chart_left:
        st.markdown("### Sector Exposure")
        if analysis.sector_exposure:
            sector_df_data = {
                "sector": list(analysis.sector_exposure.keys()),
                "weight": list(analysis.sector_exposure.values()),
            }
            fig_sector = px.pie(
                sector_df_data,
                names="sector",
                values="weight",
                hole=0.45,
                color_discrete_sequence=[
                    ACCENT_BLUE,
                    ACCENT_GREEN,
                    ACCENT_AMBER,
                    ACCENT_PURPLE,
                    ACCENT_ORANGE,
                    ACCENT_RED,
                    TEXT_MUTED,
                ],
            )
            fig_sector.update_layout(**PLOTLY_LAYOUT)
            fig_sector.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No sector exposure data available.")

    # Correlation Heatmap
    with chart_right:
        st.markdown("### Correlation Matrix")
        if analysis.correlation_matrix:
            symbols = list(analysis.correlation_matrix.keys())
            corr_z = []
            corr_text = []
            for sym_row in symbols:
                row_vals = []
                row_text = []
                for sym_col in symbols:
                    val = analysis.correlation_matrix[sym_row].get(sym_col, 0.0)
                    row_vals.append(val)
                    row_text.append(f"{val:.2f}")
                corr_z.append(row_vals)
                corr_text.append(row_text)

            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_z,
                    x=symbols,
                    y=symbols,
                    colorscale="RdBu_r",
                    zmid=0,
                    text=corr_text,
                    texttemplate="%{text}",
                    hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                )
            )
            fig_corr.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No correlation data available.")

    # ── Monte Carlo (progressive disclosure) ────────────────────────────
    with st.expander("Monte Carlo Simulation", expanded=False):
        mc: MonteCarloResult | None = None

        with st.spinner("Running Monte Carlo simulation..."):
            try:
                all_returns: list[np.ndarray] = []
                for h in analysis.holdings:
                    df = fetch_prices(h.symbol, period=period)
                    ret = df["Close"].pct_change().dropna().values
                    all_returns.append(ret * h.weight)

                min_len = min(len(r) for r in all_returns)
                portfolio_returns = np.sum(
                    [r[:min_len] for r in all_returns], axis=0
                )

                selected_dist = dist_options[dist_choice]
                mc_kwargs: dict = {
                    "initial_value": 100000,
                    "n_simulations": n_sims,
                    "time_horizon_days": horizon,
                }
                if selected_dist is None:
                    mc_kwargs["auto_fit"] = True
                else:
                    mc_kwargs["distribution"] = selected_dist

                mc = run_monte_carlo(portfolio_returns, **mc_kwargs)
            except DataError as exc:
                st.error(f"Data error during Monte Carlo: {exc}")
            except NetworkError as exc:
                st.error(f"Network error during Monte Carlo: {exc}")
            except ComputeError as exc:
                st.error(f"Computation error during Monte Carlo: {exc}")

        if mc is not None:
            # Fan Chart
            fig_mc = go.Figure()

            # Sample paths (up to 50, semi-transparent)
            paths_to_show = mc.simulated_paths[:50]
            for i, path in enumerate(paths_to_show):
                fig_mc.add_trace(
                    go.Scatter(
                        y=path,
                        mode="lines",
                        line={"color": ACCENT_BLUE, "width": 0.5},
                        opacity=0.1,
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"sim_{i}",
                    )
                )

            # Percentile bands
            paths_array = np.array(mc.simulated_paths)
            percentile_config = [
                (5, ACCENT_RED, "5th"),
                (25, ACCENT_AMBER, "25th"),
                (50, TEXT_PRIMARY, "Median"),
                (75, ACCENT_AMBER, "75th"),
                (95, ACCENT_GREEN, "95th"),
            ]
            for pctile, color, name in percentile_config:
                band = np.percentile(paths_array, pctile, axis=0)
                fig_mc.add_trace(
                    go.Scatter(
                        y=band,
                        name=name,
                        mode="lines",
                        line={
                            "color": color,
                            "width": 2 if pctile == 50 else 1,
                            "dash": "solid" if pctile == 50 else "dash",
                        },
                    )
                )

            # Initial value reference line
            fig_mc.add_hline(
                y=mc.initial_value,
                line_dash="dash",
                line_color=TEXT_MUTED,
                annotation_text="Initial",
                annotation_position="bottom right",
            )

            fig_mc.update_layout(
                **PLOTLY_LAYOUT,
                title="Monte Carlo Fan Chart",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # ── MC Statistics Row ─────────────────────────────────────
            mc_cols = st.columns(4)

            with mc_cols[0]:
                render_metric_card(
                    "Mean Final Value",
                    format_currency(mc.mean_final),
                    directional_color(mc.mean_final - mc.initial_value),
                    glossary_key="Monte Carlo",
                )
            with mc_cols[1]:
                render_metric_card(
                    "Prob of Loss",
                    format_pct(mc.prob_loss),
                    ACCENT_RED,
                    glossary_key="Tail Risk",
                )
            with mc_cols[2]:
                render_metric_card(
                    "Prob 10%+ Gain",
                    format_pct(mc.prob_gain_10pct),
                    ACCENT_GREEN,
                )
            with mc_cols[3]:
                render_metric_card(
                    "5th Percentile",
                    format_currency(mc.percentile_5),
                    ACCENT_AMBER,
                    glossary_key="VaR",
                )

            # ── Distribution Info ─────────────────────────────────────
            dist_info_cols = st.columns(3)
            with dist_info_cols[0]:
                render_metric_card(
                    "Distribution", mc.distribution_used.value.replace("_", " ").title(), ACCENT_PURPLE,
                )
            with dist_info_cols[1]:
                render_metric_card("Skewness", format_number(mc.skewness, 3), ACCENT_BLUE)
            with dist_info_cols[2]:
                render_metric_card("Kurtosis", format_number(mc.kurtosis, 3), ACCENT_AMBER)

            if mc.distribution_params is not None:
                mc_params = mc.distribution_params
                param_parts = [f"Mean: {mc_params.mean:.6f}", f"Std: {mc_params.std:.6f}"]
                if mc_params.df is not None:
                    param_parts.append(f"DF: {mc_params.df:.2f}")
                if mc_params.skew_param is not None:
                    param_parts.append(f"Skew Param: {mc_params.skew_param:.3f}")
                if mc_params.johnson_a is not None:
                    param_parts.append(f"a={mc_params.johnson_a:.3f}, b={mc_params.johnson_b:.3f}")
                st.caption(f"Fitted params: {' | '.join(param_parts)}")

    # ── Portfolio Optimization (progressive disclosure) ────────────────
    with st.expander("Portfolio Optimization", expanded=False):
        opt_cols = st.columns([2, 1])
        with opt_cols[0]:
            opt_method = st.selectbox(
                "Optimization Method",
                ["markowitz", "risk_parity", "min_variance"],
                format_func=lambda m: {
                    "markowitz": "Max Sharpe (Markowitz)",
                    "risk_parity": "Risk Parity",
                    "min_variance": "Min Variance",
                }[m],
                key="portfolio_opt_method",
                help="Algorithm to find optimal weights for your holdings",
            )
        with opt_cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            run_optimize = st.button("Optimize", type="primary", key="portfolio_opt_run")

        # ── Optimization Constraints ──────────────────────────────────
        constraints: OptimizationConstraints | None = None
        with st.expander("Optimization Constraints"):
            st.caption("Per-asset weight bounds")
            asset_constraints: list[AssetConstraint] = []
            for h in analysis.holdings:
                ac_cols = st.columns([2, 1, 1])
                with ac_cols[0]:
                    st.markdown(f"**{h.symbol}**")
                with ac_cols[1]:
                    ac_min = st.number_input(
                        "Min %",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"opt_min_{h.symbol}",
                        help="Minimum weight this asset must have in the portfolio",
                    )
                with ac_cols[2]:
                    ac_max = st.number_input(
                        "Max %",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0,
                        step=1.0,
                        key=f"opt_max_{h.symbol}",
                        help="Maximum weight this asset can have in the portfolio",
                    )
                asset_constraints.append(
                    AssetConstraint(
                        symbol=h.symbol,
                        min_weight=ac_min / 100.0,
                        max_weight=ac_max / 100.0,
                    )
                )

            st.caption("Sector constraints")
            available_sectors = list(analysis.sector_exposure.keys()) if analysis.sector_exposure else []
            selected_sectors = st.multiselect(
                "Sectors to constrain",
                options=available_sectors,
                key="opt_sector_select",
                help="Pick sectors to set a maximum weight limit on",
            )
            group_constraints: list[GroupConstraint] = []
            for sector in selected_sectors:
                sector_max = st.slider(
                    f"Max weight for {sector}",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    key=f"opt_sector_max_{sector}",
                    help=f"Cap how much of the portfolio can be in {sector}",
                )
                # Find symbols in this sector
                sector_symbols = [
                    h.symbol for h in analysis.holdings
                    if analysis.sector_exposure.get(sector, 0) > 0
                ]
                if sector_symbols:
                    group_constraints.append(
                        GroupConstraint(
                            name=sector,
                            symbols=sector_symbols,
                            max_weight=sector_max / 100.0,
                        )
                    )

            turnover_limit = st.slider(
                "Turnover Limit",
                min_value=0,
                max_value=100,
                value=100,
                step=5,
                key="opt_turnover",
                help="Maximum total weight change from current portfolio (0-100%)",
            )

            te_target = st.slider(
                "Tracking Error Target",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                format="%.1f%%",
                key="opt_te_target",
                help="Target tracking error vs benchmark (0 = no constraint)",
            )

            # Build constraints object if any constraints are non-default
            has_asset_bounds = any(
                ac.min_weight > 0.0 or ac.max_weight < 1.0 for ac in asset_constraints
            )
            has_turnover = turnover_limit < 100
            has_te = te_target > 0.0

            if has_asset_bounds or group_constraints or has_turnover or has_te:
                constraints = OptimizationConstraints(
                    asset_constraints=asset_constraints if has_asset_bounds else [],
                    group_constraints=group_constraints,
                    max_turnover=turnover_limit / 100.0 if has_turnover else None,
                    current_weights=holdings if has_turnover else {},
                    tracking_error_target=te_target / 100.0 if has_te else None,
                )

        if run_optimize:
            with st.spinner("Optimizing portfolio..."):
                try:
                    from quant_tools.portfolio.optimizer import PortfolioOptimizer

                    # Build returns DataFrame from holdings
                    returns_frames = {}
                    for h in analysis.holdings:
                        df_ret = fetch_prices(h.symbol, period=period)
                        returns_frames[h.symbol] = df_ret["Close"].pct_change().dropna()

                    returns_df = pd.DataFrame(returns_frames).dropna()

                    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)

                    # Call the specific method with constraints support
                    opt_methods = {
                        "markowitz": optimizer.optimize_markowitz,
                        "risk_parity": optimizer.optimize_risk_parity,
                        "min_variance": optimizer.optimize_min_variance,
                    }
                    opt_fn = opt_methods[opt_method]
                    opt_result = opt_fn(returns_df, constraints=constraints)

                    # Store returns_df and optimizer in session state for frontier
                    st.session_state["_opt_returns_df"] = returns_df
                    st.session_state["_opt_optimizer"] = optimizer
                    st.session_state["_opt_constraints"] = constraints
                    st.session_state["_opt_result"] = opt_result

                    # Display optimal weights
                    opt_kpi = st.columns(3)
                    with opt_kpi[0]:
                        render_metric_card(
                            "Expected Return",
                            format_pct(opt_result.expected_return),
                            directional_color(opt_result.expected_return),
                            delta=opt_result.expected_return,
                            delta_suffix="%",
                        )
                    with opt_kpi[1]:
                        render_metric_card(
                            "Volatility",
                            format_pct(opt_result.volatility),
                            ACCENT_AMBER,
                            glossary_key="Annualized Volatility",
                        )
                    with opt_kpi[2]:
                        render_metric_card(
                            "Sharpe Ratio",
                            format_number(opt_result.sharpe_ratio),
                            directional_color(opt_result.sharpe_ratio),
                            glossary_key="Sharpe Ratio",
                            delta=opt_result.sharpe_ratio,
                        )

                    if opt_result.tracking_error is not None:
                        st.caption(f"Tracking Error: {format_pct(opt_result.tracking_error)}")

                    st.markdown("**Optimal Weights:**")
                    weight_data = [
                        {"Symbol": sym, "Current": f"{holdings.get(sym, 0):.1%}", "Optimal": f"{w:.1%}"}
                        for sym, w in sorted(opt_result.weights.items(), key=lambda x: -x[1])
                    ]
                    st.dataframe(weight_data, use_container_width=True, hide_index=True)

                except Exception as exc:
                    st.error(f"Optimization failed: {exc}")

        # ── Efficient Frontier ────────────────────────────────────────
        show_frontier = st.button(
            "Show Efficient Frontier", key="portfolio_frontier_run",
        )
        if show_frontier:
            with st.spinner("Computing efficient frontier..."):
                try:
                    from quant_tools.portfolio.optimizer import PortfolioOptimizer

                    # Reuse stored data or rebuild
                    if "_opt_returns_df" in st.session_state:
                        ef_returns_df = st.session_state["_opt_returns_df"]
                        ef_optimizer = st.session_state["_opt_optimizer"]
                        ef_constraints = st.session_state.get("_opt_constraints")
                    else:
                        ef_frames = {}
                        for h in analysis.holdings:
                            df_ef = fetch_prices(h.symbol, period=period)
                            ef_frames[h.symbol] = df_ef["Close"].pct_change().dropna()
                        ef_returns_df = pd.DataFrame(ef_frames).dropna()
                        ef_optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
                        ef_constraints = constraints

                    frontier = ef_optimizer.optimize_efficient_frontier(
                        ef_returns_df, n_points=30, constraints=ef_constraints,
                    )

                    if frontier.points:
                        ef_vols = [p.volatility for p in frontier.points]
                        ef_rets = [p.expected_return for p in frontier.points]
                        ef_sharpes = [p.sharpe_ratio for p in frontier.points]

                        fig_ef = go.Figure()

                        # Frontier curve colored by Sharpe
                        fig_ef.add_trace(
                            go.Scatter(
                                x=ef_vols,
                                y=ef_rets,
                                mode="markers+lines",
                                marker={
                                    "color": ef_sharpes,
                                    "colorscale": "Viridis",
                                    "size": 6,
                                    "colorbar": {"title": "Sharpe"},
                                },
                                line={"color": ACCENT_BLUE, "width": 1},
                                name="Efficient Frontier",
                                hovertemplate=(
                                    "Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>"
                                    "Sharpe: %{marker.color:.2f}<extra></extra>"
                                ),
                            )
                        )

                        # Max Sharpe point (star)
                        if frontier.max_sharpe_point:
                            msp = frontier.max_sharpe_point
                            fig_ef.add_trace(
                                go.Scatter(
                                    x=[msp.volatility],
                                    y=[msp.expected_return],
                                    mode="markers",
                                    marker={"symbol": "star", "size": 16, "color": ACCENT_GREEN},
                                    name=f"Max Sharpe ({msp.sharpe_ratio:.2f})",
                                )
                            )

                        # Min Variance point (diamond)
                        if frontier.min_variance_point:
                            mvp = frontier.min_variance_point
                            fig_ef.add_trace(
                                go.Scatter(
                                    x=[mvp.volatility],
                                    y=[mvp.expected_return],
                                    mode="markers",
                                    marker={"symbol": "diamond", "size": 14, "color": ACCENT_AMBER},
                                    name=f"Min Variance ({mvp.sharpe_ratio:.2f})",
                                )
                            )

                        # Current portfolio position
                        fig_ef.add_trace(
                            go.Scatter(
                                x=[analysis.volatility],
                                y=[analysis.annualized_return],
                                mode="markers",
                                marker={"symbol": "circle", "size": 12, "color": ACCENT_RED},
                                name="Current Portfolio",
                            )
                        )

                        fig_ef.update_layout(
                            **PLOTLY_LAYOUT,
                            title="Efficient Frontier",
                            xaxis_title="Volatility",
                            yaxis_title="Expected Return",
                            xaxis_tickformat=".1%",
                            yaxis_tickformat=".1%",
                        )
                        st.plotly_chart(fig_ef, use_container_width=True)
                    else:
                        st.warning("Could not compute efficient frontier (no feasible points).")
                except Exception as exc:
                    st.error(f"Efficient frontier failed: {exc}")

    # ── Cross-Asset Correlations (progressive disclosure) ─────────────
    with st.expander("Cross-Asset Correlations", expanded=False):
        window_options = {"21d (~1mo)": 21, "63d (~3mo)": 63, "126d (~6mo)": 126, "252d (~1yr)": 252}
        ca_window_label = st.selectbox(
            "Rolling Window",
            options=list(window_options.keys()),
            index=1,
            key="portfolio_ca_window",
            help="Window size for rolling correlation calculation",
        )
        ca_window = window_options[ca_window_label]

        ca_symbols = [h.symbol for h in analysis.holdings]
        if len(ca_symbols) >= 2:
            with st.spinner("Computing cross-asset correlations..."):
                try:
                    ca_result = calculate_cross_asset_correlations(
                        symbols=ca_symbols, period=period,
                    )

                    ca_left, ca_right = st.columns(2)

                    # Cross-asset heatmap
                    with ca_left:
                        st.markdown("**Correlation Matrix**")
                        if ca_result.correlation_matrix is not None:
                            ca_syms = ca_result.symbols
                            ca_z = ca_result.correlation_matrix.tolist()
                            ca_text = [
                                [f"{v:.2f}" for v in row] for row in ca_z
                            ]
                            fig_ca = go.Figure(
                                data=go.Heatmap(
                                    z=ca_z,
                                    x=ca_syms,
                                    y=ca_syms,
                                    colorscale="RdBu_r",
                                    zmid=0,
                                    text=ca_text,
                                    texttemplate="%{text}",
                                    hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                                )
                            )
                            fig_ca.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(fig_ca, use_container_width=True)
                        else:
                            st.info("Insufficient data for cross-asset correlation matrix.")

                    # Rolling correlation for top 2 holdings
                    with ca_right:
                        st.markdown("**Rolling Correlation (Top 2 Holdings)**")
                        top2 = sorted(analysis.holdings, key=lambda h: h.weight, reverse=True)[:2]
                        if len(top2) == 2:
                            try:
                                ret_a_df = fetch_prices(top2[0].symbol, period=period)
                                ret_b_df = fetch_prices(top2[1].symbol, period=period)
                                ret_a = ret_a_df["Close"].pct_change().dropna().values
                                ret_b = ret_b_df["Close"].pct_change().dropna().values

                                roll_corrs, roll_idx = calculate_rolling_correlations(
                                    ret_a, ret_b, window=ca_window,
                                )

                                if roll_corrs:
                                    fig_roll = go.Figure()
                                    fig_roll.add_trace(
                                        go.Scatter(
                                            x=list(range(len(roll_corrs))),
                                            y=roll_corrs,
                                            mode="lines",
                                            line={"color": ACCENT_BLUE, "width": 2},
                                            name=f"{top2[0].symbol} vs {top2[1].symbol}",
                                        )
                                    )
                                    fig_roll.add_hline(
                                        y=0, line_dash="dash", line_color=TEXT_MUTED,
                                    )
                                    fig_roll.update_layout(
                                        **PLOTLY_LAYOUT,
                                        title=f"{ca_window}d Rolling Correlation",
                                        xaxis_title="Trading Days",
                                        yaxis_title="Correlation",
                                        yaxis_range=[-1, 1],
                                    )
                                    st.plotly_chart(fig_roll, use_container_width=True)
                                else:
                                    st.info(
                                        f"Not enough data for {ca_window}-day rolling correlation."
                                    )
                            except (DataError, NetworkError) as exc:
                                st.warning(f"Could not compute rolling correlation: {exc}")
                        else:
                            st.info("Need at least 2 holdings for rolling correlation.")
                except Exception as exc:
                    st.warning(f"Cross-asset correlation error: {exc}")
        else:
            st.info("Need at least 2 holdings for cross-asset correlation analysis.")

    # ── News Section ──────────────────────────────────────────────────
    top_symbols = sorted(
        analysis.holdings, key=lambda h: h.weight, reverse=True
    )[:3]
    render_news_section([h.symbol for h in top_symbols])
