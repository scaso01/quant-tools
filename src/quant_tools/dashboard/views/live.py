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


def _quote(symbol: str) -> dict[str, object]:
    """Fetch the latest daily bar for *symbol* as a watchlist row.

    Returns a row with an ``error`` key when the symbol cannot be priced, so a
    single bad ticker degrades to one bad row rather than an empty table.
    """
    import numpy as np

    from quant_tools.data.market_data import fetch_prices
    from quant_tools.errors import ComputeError, DataError, NetworkError

    try:
        df = fetch_prices(symbol, period="5d")
    except (DataError, NetworkError, ComputeError) as exc:
        return {"Symbol": symbol, "error": str(exc)}

    if df.empty or "Close" not in df:
        return {"Symbol": symbol, "error": "no data"}

    close = df["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    change = (last - prev) / prev if prev else 0.0

    volume = float(df["Volume"].iloc[-1]) if "Volume" in df else 0.0
    # VWAP over the loaded window, typical price weighted by volume. Falls back
    # to the mean close when the feed carries no volume.
    if "Volume" in df and float(df["Volume"].sum()) > 0:
        if {"High", "Low"}.issubset(df.columns):
            typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        else:
            typical = close
        vwap = float(np.average(typical, weights=df["Volume"]))
    else:
        vwap = float(close.mean())

    return {
        "Symbol": symbol,
        "Price": last,
        "Change": change,
        "Volume": volume,
        "VWAP": vwap,
    }


def render() -> None:
    """Render the Live Monitor page."""
    import pandas as pd
    import streamlit as st

    from quant_tools.dashboard.core.colors import ACCENT_BLUE
    from quant_tools.dashboard.core.helpers import (
        directional_color,
        format_number,
        format_pct,
        render_metric_card,
    )
    from quant_tools.dashboard.core.page_help import render_page_help

    st.header("Live Monitor")
    st.markdown("Real-time price feeds and watchlist tracking.")

    render_page_help("live")

    # \u2500\u2500 Sidebar controls \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with st.sidebar:
        st.markdown("### Watchlist")

        raw_symbols = st.text_area(
            "Symbols (comma-separated)",
            value=", ".join(DEFAULT_WATCHLIST),
            height=80,
            key="live_watchlist",
            help="Symbols to track, separated by commas, spaces or newlines",
        )

        refresh_seconds = st.slider(
            "Refresh (seconds)",
            min_value=5,
            max_value=300,
            value=5,
            step=5,
            key="live_refresh",
            help="How often the Refresh button re-reads prices",
        )

        refresh_clicked = st.button(
            "Refresh Now",
            type="primary",
            key="live_refresh_now",
            use_container_width=True,
        )

    symbols = parse_watchlist(raw_symbols)

    if not symbols:
        st.warning("Add symbols to the watchlist to start tracking prices.")
        return

    # Nothing is fetched until asked, matching Scanner's "Run Scan". Navigating
    # to a page should never block on the network.
    if refresh_clicked:
        st.session_state["live_has_run"] = True
        st.cache_data.clear()
    if not st.session_state.get("live_has_run"):
        # Naming the parsed symbols confirms the watchlist was read as intended
        # before any network call is made.
        st.info(
            f"Tracking {len(symbols)} symbols: {', '.join(symbols)}. "
            "Press **Refresh Now** to load current quotes."
        )
        return

    # \u2500\u2500 Quotes \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with st.spinner(f"Loading {len(symbols)} symbols..."):
        rows = [_quote(symbol) for symbol in symbols]

    priced = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    if failed:
        st.warning(
            "Could not price: "
            + ", ".join(f"{r['Symbol']} ({r['error']})" for r in failed)
        )

    if not priced:
        st.warning("None of the watchlist symbols could be priced.")
        return

    # \u2500\u2500 Movers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ranked = sorted(priced, key=lambda r: float(r["Change"]), reverse=True)
    best, worst = ranked[0], ranked[-1]
    advancing = sum(1 for r in priced if float(r["Change"]) > 0)

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Tracking", str(len(priced)), ACCENT_BLUE)
    with cols[1]:
        render_metric_card(
            "Top Gainer",
            f"{best['Symbol']} {format_pct(float(best['Change']))}",
            directional_color(float(best["Change"])),
        )
    with cols[2]:
        render_metric_card(
            "Top Loser",
            f"{worst['Symbol']} {format_pct(float(worst['Change']))}",
            directional_color(float(worst["Change"])),
        )
    with cols[3]:
        render_metric_card(
            "Advancing",
            f"{advancing}/{len(priced)}",
            directional_color(advancing - len(priced) / 2),
        )

    st.divider()

    # \u2500\u2500 Watchlist table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown("### Watchlist")
    table = pd.DataFrame(priced)[["Symbol", "Price", "Change", "Volume", "VWAP"]]
    st.dataframe(
        table.style.format(
            {
                "Price": "${:,.2f}",
                "Change": "{:+.2%}",
                "Volume": "{:,.0f}",
                "VWAP": "${:,.2f}",
            }
        ).map(
            lambda v: f"color: {directional_color(v)}", subset=["Change"]
        ),
        use_container_width=True,
    )

    st.caption(
        f"Prices are end-of-day bars from the cached data layer, not a live "
        f"stream. Press Refresh Now to re-read them, at most every "
        f"{format_number(refresh_seconds, decimals=0)} seconds."
    )
