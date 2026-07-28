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


def render() -> None:
    """Render the Rolling Analytics dashboard page.

    Uses streamlit (imported at call time to avoid import-time side effects).
    """
    import streamlit as st

    from quant_tools.dashboard.core.page_help import render_page_help

    st.header("Rolling Analytics")
    st.markdown("Analyze rolling-window risk and return metrics over time.")

    render_page_help("rolling")

    st.markdown(
        '<div class="coming-soon-card">'
        '<div class="coming-soon-badge">In Development</div>'
        '<div class="coming-soon-title">Rolling Analytics</div>'
        '<div class="coming-soon-desc">'
        'Track how risk and return metrics change over time with rolling windows. '
        'Includes rolling Sharpe ratio, rolling volatility, and rolling beta vs. a benchmark.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This page will compute rolling-window metrics from your portfolio data. "
        "Start by analyzing a portfolio in the **Portfolio Analyzer**."
    )

    if st.button("Go to Portfolio Analyzer \u2192", type="primary"):
        st.session_state["active_page"] = "\U0001f4bc Portfolio Analyzer"
        st.query_params["page"] = "\U0001f4bc Portfolio Analyzer"
        st.rerun()
