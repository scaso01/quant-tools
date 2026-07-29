"""Risk Dashboard page — portfolio risk metrics at a glance.

Shows VaR, CVaR, concentration risk, correlation heatmap, and
drawdown analysis. Gracefully degrades when optional risk modules
are not available.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Risk computation helpers (pure numpy)
# =============================================================================


def compute_var(
    returns: pd.Series, confidence: float = 0.95, method: str = "historical"
) -> float:
    """Value at Risk at given confidence level.

    Parameters
    ----------
    returns:
        Daily returns series.
    confidence:
        Confidence level (e.g., 0.95 for 95%).
    method:
        "historical" (percentile) or "parametric" (normal assumption).

    Returns
    -------
    VaR as a positive number (loss).
    """
    if returns.empty:
        return 0.0

    if method == "parametric":
        from scipy.stats import norm  # type: ignore[import-untyped]
        z = norm.ppf(1 - confidence)
        return float(-(returns.mean() + z * returns.std()))
    else:
        # Historical percentile
        return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    if returns.empty:
        return 0.0
    var = compute_var(returns, confidence, method="historical")
    tail = returns[returns <= -var]
    if tail.empty:
        return var
    return float(-tail.mean())


def compute_concentration(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index (HHI) for portfolio concentration.

    Returns value between 0 (perfect diversification) and 1 (single asset).
    """
    if not weights:
        return 0.0
    w = np.array(list(weights.values()))
    w = w / w.sum() if w.sum() > 0 else w
    return float(np.sum(w ** 2))


# =============================================================================
# Dashboard render
# =============================================================================


DEFAULT_HOLDINGS = {"AAPL": 0.3, "MSFT": 0.2, "BND": 0.3, "GLD": 0.2}

VAR_METHODS = {"Historical": "historical", "Parametric": "parametric"}


def _load_holdings() -> dict[str, float]:
    """Read holdings from the Portfolio Analyzer, falling back to a default.

    The two pages share ``portfolio_holdings_json`` so that analysing a
    portfolio on one page carries over to the risk view on the other.
    """
    import json

    import streamlit as st

    raw = st.session_state.get("portfolio_holdings_json")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return dict(DEFAULT_HOLDINGS)
        if isinstance(parsed, dict) and parsed:
            return {
                str(k): float(v)
                for k, v in parsed.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    return dict(DEFAULT_HOLDINGS)


def render() -> None:
    """Render the Risk Dashboard page."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    from quant_tools.backtest.metrics import calculate_max_drawdown
    from quant_tools.dashboard.core.colors import (
        ACCENT_AMBER,
        ACCENT_BLUE,
        ACCENT_RED,
    )
    from quant_tools.dashboard.core.date_presets import render_date_presets
    from quant_tools.dashboard.core.helpers import (
        PLOTLY_LAYOUT,
        format_number,
        format_pct,
        render_metric_card,
    )
    from quant_tools.dashboard.core.page_help import render_page_help
    from quant_tools.data.market_data import fetch_prices
    from quant_tools.errors import ComputeError, DataError, NetworkError

    st.header("Risk Dashboard")
    st.markdown("Monitor portfolio risk metrics in real time.")

    render_page_help("risk_dashboard")

    holdings = _load_holdings()

    # \u2500\u2500 Sidebar controls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with st.sidebar:
        st.markdown("### Risk Settings")

        confidence = st.slider(
            "VaR Confidence",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f",
            key="risk_var_confidence",
            help="Confidence level for Value at Risk, e.g. 0.95 = 95%",
        )

        method_label = st.selectbox(
            "VaR Method",
            list(VAR_METHODS.keys()),
            key="risk_var_method",
            help=(
                "Historical uses the observed return distribution, "
                "Parametric assumes returns are normal"
            ),
        )

        start_date, end_date = render_date_presets(
            "risk", default="1Y", sidebar=False,
        )

        st.caption(
            f"{len(holdings)} holdings loaded from the Portfolio Analyzer."
        )

        run_analysis = st.button(
            "Analyze Risk", type="primary", key="risk_run", use_container_width=True
        )

    # Nothing is fetched until asked, matching Scanner's "Run Scan" and
    # Portfolio's "Analyze Portfolio". Navigating to a page should never block
    # on the network.
    if run_analysis:
        st.session_state["risk_has_run"] = True
    if not st.session_state.get("risk_has_run"):
        st.info(
            f"Ready to analyze {len(holdings)} holdings. Press "
            "**Analyze Risk** to compute VaR, drawdown and concentration."
        )
        return

    # \u2500\u2500 Load prices and build the portfolio return series \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    try:
        with st.spinner("Loading portfolio prices..."):
            frames: dict[str, pd.Series] = {}
            missing: list[str] = []
            for symbol in holdings:
                try:
                    df = fetch_prices(
                        symbol, start=str(start_date), end=str(end_date)
                    )
                    frames[symbol] = df["Close"].pct_change().dropna()
                except (DataError, ComputeError):
                    missing.append(symbol)
    except NetworkError as exc:
        st.error(f"Network error: {exc}")
        return

    if missing:
        st.warning(f"Skipped, no price data: {', '.join(missing)}")

    if not frames:
        st.warning(
            "No price data available for these holdings, so risk metrics "
            "cannot be computed."
        )
        return

    returns_df = pd.DataFrame(frames).dropna()
    if returns_df.empty:
        st.warning("Holdings have no overlapping history over this date range.")
        return

    # Re-weight over whatever actually loaded so the weights still sum to 1.
    used_weights = {s: holdings[s] for s in returns_df.columns}
    total_weight = sum(used_weights.values())
    if total_weight <= 0:
        st.warning("Holding weights sum to zero, cannot compute risk metrics.")
        return
    norm_weights = {s: w / total_weight for s, w in used_weights.items()}

    portfolio_returns = sum(
        returns_df[s] * w for s, w in norm_weights.items()
    )

    var = compute_var(
        portfolio_returns, confidence=confidence, method=VAR_METHODS[method_label]
    )
    cvar = compute_cvar(portfolio_returns, confidence=confidence)
    concentration = compute_concentration(used_weights)

    equity_curve = (1.0 + portfolio_returns).cumprod().to_numpy()
    max_dd, dd_duration = calculate_max_drawdown(equity_curve)

    # \u2500\u2500 Metric cards \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    cols = st.columns(4)
    with cols[0]:
        render_metric_card(
            f"VaR ({confidence:.0%}, 1d)",
            format_pct(var),
            ACCENT_RED,
            glossary_key="VaR",
        )
    with cols[1]:
        render_metric_card(
            f"CVaR ({confidence:.0%}, 1d)",
            format_pct(cvar),
            ACCENT_RED,
            glossary_key="CVaR",
        )
    with cols[2]:
        render_metric_card(
            "Concentration (HHI)",
            format_number(concentration, decimals=3),
            ACCENT_AMBER,
        )
    with cols[3]:
        render_metric_card(
            "Max Drawdown",
            format_pct(max_dd),
            ACCENT_RED,
            glossary_key="Max Drawdown",
        )

    st.caption(
        f"Longest drawdown lasted {dd_duration} trading days. "
        f"VaR method: {method_label}."
    )

    st.divider()

    # \u2500\u2500 Per-holding risk table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown("### Risk by Holding")
    rows = [
        {
            "Symbol": symbol,
            "Weight": norm_weights[symbol],
            "Volatility (ann.)": float(returns_df[symbol].std() * np.sqrt(252)),
            f"VaR ({confidence:.0%})": compute_var(
                returns_df[symbol],
                confidence=confidence,
                method=VAR_METHODS[method_label],
            ),
            f"CVaR ({confidence:.0%})": compute_cvar(
                returns_df[symbol], confidence=confidence
            ),
        }
        for symbol in returns_df.columns
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "Weight": "{:.1%}",
                "Volatility (ann.)": "{:.2%}",
                f"VaR ({confidence:.0%})": "{:.2%}",
                f"CVaR ({confidence:.0%})": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    # \u2500\u2500 Drawdown chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown("### Portfolio Drawdown")
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = np.where(
        running_max > 0, (equity_curve - running_max) / running_max, 0.0
    )
    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=returns_df.index,
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line={"color": ACCENT_RED, "width": 1},
            name="Drawdown",
        )
    )
    dd_fig.update_layout(
        **PLOTLY_LAYOUT, title="Drawdown from running peak",
        xaxis_title="Date", yaxis_title="Drawdown",
    )
    dd_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(dd_fig, use_container_width=True)

    # \u2500\u2500 Correlation heatmap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if len(returns_df.columns) > 1:
        st.markdown("### Correlation")
        corr = returns_df.corr()
        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
            )
        )
        heat.update_layout(**PLOTLY_LAYOUT, title="Holding correlation")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info(
            "Correlation needs at least two holdings with overlapping history."
        )
