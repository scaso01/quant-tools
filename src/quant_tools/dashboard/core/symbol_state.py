"""Global symbol state management for the Streamlit dashboard.

Manages a single "active symbol" that persists across pages via
session state and URL query parameters, along with recent symbols
and user favorites.
"""
from __future__ import annotations

import streamlit as st


def init_global_symbol() -> None:
    """Initialize global symbol state from URL params.

    Call once in ``app.py`` after CSS loading.  Reads the ``symbol``
    query parameter and seeds session state if not already set.
    """
    if "global_symbol" not in st.session_state:
        qp_symbol = st.query_params.get("symbol", "")
        st.session_state["global_symbol"] = qp_symbol if qp_symbol else None

    if "recent_symbols" not in st.session_state:
        st.session_state["recent_symbols"] = []

    if "favorite_symbols" not in st.session_state:
        st.session_state["favorite_symbols"] = []


def get_active_symbol() -> str | None:
    """Return the current global symbol, or ``None`` if unset."""
    return st.session_state.get("global_symbol")


def set_active_symbol(symbol: str) -> None:
    """Set the active symbol, sync to URL, and append to recents.

    Args:
        symbol: Ticker string (e.g. ``"AAPL"``).
    """
    upper = symbol.strip().upper()
    if not upper:
        return

    st.session_state["global_symbol"] = upper
    st.query_params["symbol"] = upper

    # Append to recents (deduped, max 10, most-recent first)
    recents: list[str] = st.session_state.get("recent_symbols", [])
    if upper in recents:
        recents.remove(upper)
    recents.insert(0, upper)
    st.session_state["recent_symbols"] = recents[:10]


def get_recent_symbols() -> list[str]:
    """Return up to 10 most-recently used symbols (newest first)."""
    return list(st.session_state.get("recent_symbols", []))


def get_favorites() -> list[str]:
    """Return the user's favorited symbols."""
    return list(st.session_state.get("favorite_symbols", []))


def toggle_favorite(symbol: str) -> None:
    """Add or remove *symbol* from favorites.

    Args:
        symbol: Ticker string to toggle.
    """
    upper = symbol.strip().upper()
    if not upper:
        return

    favs: list[str] = st.session_state.get("favorite_symbols", [])
    if upper in favs:
        favs.remove(upper)
    else:
        favs.append(upper)
    st.session_state["favorite_symbols"] = favs
