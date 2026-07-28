"""Live Monitor page — real-time price feed and watchlist.

Displays streaming prices, a configurable watchlist, and basic
real-time analytics (VWAP, spread, volume).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Watchlist helpers
# =============================================================================

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def parse_watchlist(text: str) -> list[str]:
    """Parse a comma/newline-separated watchlist string into symbols.

    Parameters
    ----------
    text:
        Raw text with symbols separated by commas, spaces, or newlines.

    Returns
    -------
    List of uppercase, deduplicated symbols.
    """
    if not text or not text.strip():
        return []

    # Split on commas, spaces, newlines
    import re
    raw = re.split(r"[,\s\n]+", text.strip())
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        sym = item.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            result.append(sym)
    return result


# =============================================================================
# Dashboard render
# =============================================================================


def render() -> None:
    """Render the Live Monitor page."""
    import streamlit as st

    from quant_tools.dashboard.core.page_help import render_page_help

    st.header("Live Monitor")
    st.markdown("Real-time price feeds and watchlist tracking.")

    render_page_help("live")

    st.markdown(
        '<div class="coming-soon-card">'
        '<div class="coming-soon-badge">In Development</div>'
        '<div class="coming-soon-title">Live Monitor</div>'
        '<div class="coming-soon-desc">'
        'Track real-time prices for your watchlist with auto-refreshing feeds. '
        'Includes price, change, volume, and VWAP. Requires WebSocket integration.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This page will show live price data once WebSocket feeds are connected. "
        "In the meantime, check current prices in the **Market Scanner**."
    )

    if st.button("Go to Market Scanner \u2192", type="primary"):
        st.session_state["active_page"] = "\U0001f4e1 Market Scanner"
        st.query_params["page"] = "\U0001f4e1 Market Scanner"
        st.rerun()
