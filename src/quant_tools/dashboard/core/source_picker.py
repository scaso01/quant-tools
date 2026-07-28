"""Data source picker widget for dashboard pages."""

from __future__ import annotations

import streamlit as st


def render_data_source_picker(category: str, page_key: str) -> dict[str, str]:
    """Sidebar data source selector.

    Args:
        category: One of 'equity', 'crypto', 'sentiment', 'macro', or 'all'.
        page_key: Unique key prefix to avoid Streamlit widget ID collisions.

    Returns:
        Dict mapping role names to selected provider names.
    """
    sources: dict[str, str] = {}

    with st.sidebar:
        st.markdown("### Data Sources")

        if category in ("equity", "all"):
            sources["ohlcv"] = st.selectbox(
                "OHLCV Provider",
                ["yfinance", "finnhub", "stooq"],
                key=f"{page_key}_ohlcv",
            )

        if category in ("crypto", "all"):
            sources["crypto"] = st.selectbox(
                "Crypto Provider",
                ["ccxt", "binance", "coingecko"],
                key=f"{page_key}_crypto",
            )

        if category in ("sentiment", "all"):
            selected = st.multiselect(
                "Sentiment Sources",
                ["reddit", "finbert", "gdelt", "google_trends", "fear_greed"],
                default=["reddit"],
                key=f"{page_key}_sentiment",
            )
            sources["sentiment"] = ",".join(selected)

        if category in ("macro", "all"):
            sources["macro"] = st.selectbox(
                "Macro Provider",
                ["fred", "worldbank", "dbnomics", "ecb"],
                key=f"{page_key}_macro",
            )

    return sources
