"""Factors & Macro Regime view for the Streamlit dashboard."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_tools.config import cfg
from quant_tools.dashboard.core.colors import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    TEXT_MUTED,
    hex_to_rgb,
)
from quant_tools.dashboard.core.helpers import (
    PLOTLY_LAYOUT,
    directional_color,
    format_pct,
    regime_color,
    render_metric_card,
    render_news_section,
)
from quant_tools.dashboard.core.page_help import render_page_help
from quant_tools.dashboard.core.symbol_picker import render_symbol_picker
from quant_tools.data.universe import SP500_TOP100
from quant_tools.errors import ComputeError, DataError, NetworkError
from quant_tools.factors.macro import detect_regime
from quant_tools.factors.rebalance import get_recommendations
from quant_tools.factors.scoring import calculate_factor_scores
from quant_tools.models import (
    FactorScore,
    MacroSnapshot,
    RebalanceRecommendation,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_symbols(text: str) -> list[str]:
    """Parse a text area of symbols (comma or newline separated)."""
    raw = text.replace(",", "\n").split("\n")
    symbols: list[str] = []
    seen: set[str] = set()
    for token in raw:
        sym = token.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols


def _scores_to_dataframe(scores: list[FactorScore]) -> pd.DataFrame:
    """Convert factor scores to a DataFrame for display."""
    rows = []
    for s in scores:
        rows.append(
            {
                "Rank": s.rank,
                "Symbol": s.symbol,
                "Name": s.name,
                "Sector": s.sector,
                "Value Z": s.value_z,
                "Momentum Z": s.momentum_z,
                "Quality Z": s.quality_z,
                "Composite Z": s.composite_z,
            }
        )
    return pd.DataFrame(rows)


def _picks_to_dataframe(picks: list[FactorScore]) -> pd.DataFrame:
    """Convert top picks to a DataFrame for display."""
    rows = []
    for s in picks:
        rows.append(
            {
                "Rank": s.rank,
                "Symbol": s.symbol,
                "Name": s.name,
                "Sector": s.sector,
                "Composite Z": s.composite_z,
            }
        )
    return pd.DataFrame(rows)


def _build_factor_bar_chart(scores: list[FactorScore]) -> None:
    """Render a grouped bar chart of factor z-scores for the top 10."""
    top = scores[:10]
    symbols = [s.symbol for s in top]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Value",
            x=symbols,
            y=[s.value_z for s in top],
            marker_color=ACCENT_GREEN,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Momentum",
            x=symbols,
            y=[s.momentum_z for s in top],
            marker_color=ACCENT_BLUE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Quality",
            x=symbols,
            y=[s.quality_z for s in top],
            marker_color=ACCENT_PURPLE,
        )
    )
    fig.update_layout(barmode="group", title="Factor Z-Scores (Top 10)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _build_sector_pie(scores: list[FactorScore]) -> None:
    """Render a sector distribution pie chart."""
    counts: Counter[str] = Counter()
    for s in scores:
        counts[s.sector or "Unknown"] += 1

    df = pd.DataFrame(
        {"Sector": list(counts.keys()), "Count": list(counts.values())}
    )

    fig = px.pie(
        df,
        names="Sector",
        values="Count",
        title="Sector Distribution",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _build_tilt_chart(rec: RebalanceRecommendation) -> None:
    """Render a horizontal stacked bar chart of regime tilts."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Value",
            y=["Tilt"],
            x=[rec.value_tilt],
            orientation="h",
            marker_color=ACCENT_GREEN,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Momentum",
            y=["Tilt"],
            x=[rec.momentum_tilt],
            orientation="h",
            marker_color=ACCENT_BLUE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Quality",
            y=["Tilt"],
            x=[rec.quality_tilt],
            orientation="h",
            marker_color=ACCENT_PURPLE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Defensive",
            y=["Tilt"],
            x=[rec.defensive_tilt],
            orientation="h",
            marker_color=ACCENT_AMBER,
        )
    )
    fig.update_layout(
        barmode="stack",
        title="Regime Factor Tilts",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main render ──────────────────────────────────────────────────────


def render() -> None:
    """Render the Factors & Macro Regime page."""

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>Factors &amp; Macro Regime</h1>"
        "<p>Factor-based scoring with macro regime overlay</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    render_page_help("factors")

    # ── Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Macro Regime")

        has_fred_key = bool(cfg.fred_api_key)
        detect_btn = st.button(
            "Detect Regime",
            use_container_width=True,
            disabled=not has_fred_key,
            key="factors_detect_regime",
        )
        if not has_fred_key:
            st.warning(
                "FRED API key not configured. Set FRED_API_KEY in .env "
                "to enable macro regime detection."
            )

        st.markdown("---")
        st.markdown("### Universe")

        use_sp500 = st.checkbox(
            "Use S&P 500 Top 100",
            value=True,
            key="factors_use_sp500",
            help="Score the top 100 S&P 500 stocks — uncheck to enter your own",
        )

        # Symbol picker to add symbols to the text area
        if not use_sp500:
            _picked = render_symbol_picker(
                label="Add Symbol",
                key="factors_symbol_picker",
                default=None,
                sidebar=False,  # already inside st.sidebar context
            )
            if _picked:
                _existing = st.session_state.get("factors_custom_symbols", "")
                _existing_syms = {
                    s.strip().upper()
                    for s in _existing.replace(",", "\n").split("\n")
                    if s.strip()
                }
                if _picked.upper() not in _existing_syms:
                    sep = "\n" if _existing.strip() else ""
                    st.session_state["factors_custom_symbols"] = (
                        _existing.strip() + sep + _picked.upper()
                    )
                    st.rerun()

        custom_symbols_text = st.text_area(
            "Custom Symbols (one per line or comma separated)",
            value="",
            height=120,
            disabled=use_sp500,
            key="factors_custom_symbols",
            help="Enter ticker symbols to score (e.g., AAPL, MSFT, GOOGL)",
        )

        st.markdown("---")
        st.markdown("### Factor Weights")

        value_weight = st.slider(
            "Value weight",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.01,
            key="factors_value_weight",
            help="How much weight to give Value in the composite score",
        )
        momentum_weight = st.slider(
            "Momentum weight",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.01,
            key="factors_momentum_weight",
            help="How much weight to give Momentum in the composite score",
        )
        quality_weight = st.slider(
            "Quality weight",
            min_value=0.0,
            max_value=1.0,
            value=0.34,
            step=0.01,
            key="factors_quality_weight",
            help="How much weight to give Quality in the composite score",
        )

        weight_sum = value_weight + momentum_weight + quality_weight
        if abs(weight_sum - 1.0) > 0.05:
            st.caption(
                f":orange[Weights sum to {weight_sum:.2f} — ideally should be ~1.0]"
            )

        st.markdown("---")

        top_n = st.slider(
            "Top N Results",
            min_value=5,
            max_value=30,
            value=10,
            key="factors_top_n",
            help="Number of top-ranked stocks to display",
        )

        calc_btn = st.button(
            "Calculate Factors",
            use_container_width=True,
            type="primary",
            key="factors_calculate",
        )

    # ── Detect regime ─────────────────────────────────────────────────
    if detect_btn:
        try:
            with st.spinner("Detecting macro regime from FRED data..."):
                snapshot = detect_regime()
            st.session_state["factors_snapshot"] = snapshot
        except DataError as exc:
            st.warning(f"Data issue: {exc}")
        except NetworkError as exc:
            st.error(f"Network error: {exc}")
        except ComputeError as exc:
            st.warning(f"Computation error: {exc}")

    # ── Calculate factor scores ───────────────────────────────────────
    if calc_btn:
        # Build symbol list
        if use_sp500:
            symbols = list(SP500_TOP100)
        else:
            symbols = _parse_symbols(custom_symbols_text)

        if not symbols:
            st.warning("No symbols provided. Enter symbols or select S&P 500 Top 100.")
            return

        weights = {
            "value": value_weight,
            "momentum": momentum_weight,
            "quality": quality_weight,
        }

        try:
            with st.spinner("Calculating factor scores..."):
                scores = calculate_factor_scores(symbols, weights=weights)
            st.session_state["factors_scores"] = scores
        except DataError as exc:
            st.warning(f"Data issue: {exc}")
            return
        except NetworkError as exc:
            st.error(f"Network error: {exc}")
            return
        except ComputeError as exc:
            st.warning(f"Computation error: {exc}")
            return

        # Get recommendations if regime is available
        snapshot: MacroSnapshot | None = st.session_state.get("factors_snapshot")
        if snapshot is not None:
            try:
                with st.spinner("Generating rebalance recommendations..."):
                    rec = get_recommendations(
                        snapshot.regime, scores, top_n=top_n
                    )
                st.session_state["factors_recommendation"] = rec
            except DataError as exc:
                st.warning(f"Data issue: {exc}")
            except NetworkError as exc:
                st.error(f"Network error: {exc}")
            except ComputeError as exc:
                st.warning(f"Computation error: {exc}")

    # ── Retrieve cached state ─────────────────────────────────────────
    snapshot: MacroSnapshot | None = st.session_state.get("factors_snapshot")
    scores: list[FactorScore] = st.session_state.get("factors_scores", [])
    rec: RebalanceRecommendation | None = st.session_state.get(
        "factors_recommendation"
    )

    # ── Early exit if nothing computed ────────────────────────────────
    if not scores and snapshot is None:
        st.info(
            "Configure settings in the sidebar. Click **Detect Regime** for macro "
            "analysis, or **Calculate Factors** to score the S&P 500 Top 100 "
            "with default equal weights."
        )
        return

    # ── Macro Regime Section ──────────────────────────────────────────
    if snapshot is not None:
        st.markdown("### Macro Regime")

        color = regime_color(snapshot.regime)
        rgb = hex_to_rgb(color)

        st.markdown(
            f'<span class="regime-badge" style="'
            f"display:inline-block;padding:6px 16px;border-radius:6px;"
            f"font-weight:700;font-size:1rem;"
            f"background:rgba({rgb},0.15);color:{color};"
            f'">{snapshot.regime.value.upper()}</span>',
            unsafe_allow_html=True,
        )

        # Rationale / signals
        if rec and rec.rationale:
            st.markdown(
                f'<p style="color:{TEXT_MUTED};margin-top:8px;">'
                f"{rec.rationale}</p>",
                unsafe_allow_html=True,
            )
        elif snapshot.signals:
            signal_text = " | ".join(
                f"**{k}**: {v}" for k, v in snapshot.signals.items()
            )
            st.markdown(signal_text)

        st.markdown("")

        # FRED KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            gdp_val = format_pct(snapshot.gdp_growth) if snapshot.gdp_growth is not None else "N/A"
            render_metric_card(
                "GDP Growth",
                gdp_val,
                directional_color(snapshot.gdp_growth) if snapshot.gdp_growth is not None else ACCENT_RED,
                glossary_key="GDP",
                delta=snapshot.gdp_growth,
                delta_suffix="%",
            )

        with kpi2:
            unemp_val = format_pct(snapshot.unemployment) if snapshot.unemployment is not None else "N/A"
            unemp_color = ACCENT_GREEN if snapshot.unemployment is not None and snapshot.unemployment < 0.05 else ACCENT_AMBER
            render_metric_card("Unemployment", unemp_val, unemp_color)

        with kpi3:
            cpi_val = format_pct(snapshot.cpi_yoy) if snapshot.cpi_yoy is not None else "N/A"
            cpi_color = ACCENT_GREEN if snapshot.cpi_yoy is not None and snapshot.cpi_yoy < 0.03 else ACCENT_RED
            render_metric_card("CPI YoY", cpi_val, cpi_color)

        with kpi4:
            ffr_val = format_pct(snapshot.fed_funds_rate) if snapshot.fed_funds_rate is not None else "N/A"
            render_metric_card("Fed Funds Rate", ffr_val, ACCENT_BLUE)

        with kpi5:
            spread_val = format_pct(snapshot.yield_spread_10y2y) if snapshot.yield_spread_10y2y is not None else "N/A"
            render_metric_card(
                "Yield Spread",
                spread_val,
                directional_color(snapshot.yield_spread_10y2y) if snapshot.yield_spread_10y2y is not None else ACCENT_RED,
                glossary_key="Yield Spread",
                delta=snapshot.yield_spread_10y2y,
                delta_suffix="%",
            )

        st.markdown("")

    # ── Factor Scores Section ─────────────────────────────────────────
    if not scores:
        return

    st.markdown("### Factor Scores")

    df = _scores_to_dataframe(scores)

    st.dataframe(
        df.style.format(
            {
                "Value Z": "{:+.2f}",
                "Momentum Z": "{:+.2f}",
                "Quality Z": "{:+.2f}",
                "Composite Z": "{:+.2f}",
            }
        ),
        use_container_width=True,
        height=min(400, 40 + 35 * len(scores)),
    )

    # ── Next Steps ────────────────────────────────────────────────────
    st.markdown("---")
    next_cols = st.columns([1, 1, 4])
    with next_cols[0]:
        if st.button("Get Predictions \u2192", key="factors_to_predictions"):
            st.session_state["active_page"] = "\U0001f52e Predictions"
            st.query_params["page"] = "\U0001f52e Predictions"
            st.rerun()
    with next_cols[1]:
        if st.button("Build a Portfolio \u2192", key="factors_to_portfolio"):
            st.session_state["active_page"] = "\U0001f4bc Portfolio Analyzer"
            st.query_params["page"] = "\U0001f4bc Portfolio Analyzer"
            st.rerun()

    st.markdown("")

    # ── Charts Row ────────────────────────────────────────────────────
    chart_left, chart_right = st.columns(2)

    with chart_left:
        _build_factor_bar_chart(scores)

    with chart_right:
        _build_sector_pie(scores)

    st.markdown("")

    # ── Regime Tilt Chart ─────────────────────────────────────────────
    if rec is not None:
        st.markdown("### Regime Tilts")
        _build_tilt_chart(rec)
        st.markdown("")

    # ── Top Picks Table ───────────────────────────────────────────────
    if rec is not None and rec.top_picks:
        st.markdown("### Top Picks")

        picks_df = _picks_to_dataframe(rec.top_picks)

        st.dataframe(
            picks_df.style.format({"Composite Z": "{:+.2f}"}),
            use_container_width=True,
            height=min(400, 40 + 35 * len(rec.top_picks)),
        )

        st.markdown("")

    # ── News Section ──────────────────────────────────────────────────
    top_symbols = [s.symbol for s in scores[:3]]
    render_news_section(top_symbols)
