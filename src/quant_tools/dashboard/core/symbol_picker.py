"""Reusable symbol picker widget backed by streamlit-searchbox.

Provides autocomplete search across all universes with recent-symbol
suggestions on empty query, and a star toggle for favorites.
"""
from __future__ import annotations

import streamlit as st
from streamlit_searchbox import st_searchbox

from quant_tools.dashboard.core.symbol_search import (
    SearchEntry,
    build_search_index,
    search_symbols,
)
from quant_tools.dashboard.core.symbol_state import (
    get_favorites,
    get_recent_symbols,
    set_active_symbol,
    toggle_favorite,
)


def _format_entry(entry: SearchEntry) -> tuple[str, str]:
    """Format a SearchEntry as (display_label, value) for the searchbox."""
    sector_tag = f" ({entry.sector})" if entry.sector else ""
    label = f"{entry.symbol} \u2014 {entry.name}{sector_tag}"
    return (label, entry.symbol)


def _search_callback(query: str) -> list[tuple[str, str]]:
    """Search callback for st_searchbox.

    On empty query returns formatted recent symbols; otherwise
    delegates to :func:`search_symbols`.
    """
    if not query or not query.strip():
        # Show recent symbols when the box is empty
        recents = get_recent_symbols()
        if not recents:
            return []
        index = {e.symbol: e for e in build_search_index()}
        results: list[tuple[str, str]] = []
        for sym in recents:
            entry = index.get(sym)
            if entry:
                results.append(_format_entry(entry))
            else:
                results.append((sym, sym))
        return results

    entries = search_symbols(query, limit=10)
    return [_format_entry(e) for e in entries]


def render_symbol_picker(
    label: str = "Symbol",
    key: str = "symbol_picker",
    default: str | None = None,
    sidebar: bool = True,
) -> str | None:
    """Render a searchbox symbol picker with a favorite-star toggle.

    Args:
        label: Label displayed above the searchbox.
        key: Unique Streamlit widget key.
        default: Default symbol to pre-fill (e.g. ``"AAPL"``).
        sidebar: When ``True`` (default), renders inside ``st.sidebar``.

    Returns:
        The selected symbol string, or ``None`` if nothing is selected.
    """
    container = st.sidebar if sidebar else st

    with container:
        pick_col, star_col = st.columns([5, 1])

        with pick_col:
            selected = st_searchbox(
                search_function=_search_callback,
                label=label,
                placeholder="Search by ticker or company name...",
                key=key,
                default=default,
                clear_on_submit=False,
            )

        symbol: str | None = None
        if selected:
            symbol = selected if isinstance(selected, str) else str(selected)
            set_active_symbol(symbol)

        with star_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if symbol:
                is_fav = symbol.upper() in get_favorites()
                star_label = "\u2605" if is_fav else "\u2606"
                if st.button(
                    star_label,
                    key=f"{key}_star",
                    help="Toggle favorite",
                ):
                    toggle_favorite(symbol)
                    st.rerun()

    return symbol
