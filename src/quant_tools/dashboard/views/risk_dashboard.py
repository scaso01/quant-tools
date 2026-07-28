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


def render() -> None:
    """Render the Risk Dashboard page."""
    import streamlit as st

    from quant_tools.dashboard.core.page_help import render_page_help

    st.header("Risk Dashboard")
    st.markdown("Monitor portfolio risk metrics in real time.")

    render_page_help("risk_dashboard")

    st.markdown(
        '<div class="coming-soon-card">'
        '<div class="coming-soon-badge">In Development</div>'
        '<div class="coming-soon-title">Risk Dashboard</div>'
        '<div class="coming-soon-desc">'
        'Get a comprehensive view of portfolio risk: Value at Risk (VaR), '
        'Conditional VaR, maximum drawdown, and concentration metrics. '
        'Supports both historical and parametric methods.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This page will analyze risk metrics from your portfolio. "
        "Start by building a portfolio in the **Portfolio Analyzer**."
    )

    if st.button("Go to Portfolio Analyzer \u2192", type="primary"):
        st.session_state["active_page"] = "\U0001f4bc Portfolio Analyzer"
        st.query_params["page"] = "\U0001f4bc Portfolio Analyzer"
        st.rerun()
