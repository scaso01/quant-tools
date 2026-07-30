"""Reusable date-range preset selector for dashboard views."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st


# ── Preset Definitions ──────────────────────────────────────────────────

PRESETS: dict[str, timedelta | str] = {
    "1W": timedelta(weeks=1),
    "1M": timedelta(days=30),
    "3M": timedelta(days=91),
    "6M": timedelta(days=182),
    "YTD": "YTD",
    "1Y": timedelta(days=365),
    "2Y": timedelta(days=730),
    "5Y": timedelta(days=1826),
}


def _resolve_preset(key: str) -> tuple[date, date]:
    """Resolve a preset key to a (start_date, end_date) pair.

    Args:
        key: One of the keys in :data:`PRESETS`.

    Returns:
        Tuple of (start_date, end_date).
    """
    today = date.today()
    preset_value = PRESETS[key]

    if key == "YTD":
        return date(today.year, 1, 1), today

    if isinstance(preset_value, timedelta):
        return today - preset_value, today

    # Fallback (should not happen with current PRESETS)
    return today - timedelta(days=365), today


def render_date_presets(
    key_prefix: str,
    default: str = "1Y",
    sidebar: bool = True,
) -> tuple[date, date]:
    """Render a horizontal row of date-preset buttons and return the selected range.

    Tracks the active preset in ``st.session_state["{key_prefix}_date_preset"]``.

    Args:
        key_prefix: Unique prefix for session-state keys (avoids collisions
            between different pages that both use presets).
        default: Which preset is active initially.  Must be a key in
            :data:`PRESETS`.
        sidebar: If ``True``, render inside ``st.sidebar``; otherwise render
            in the main content area.

    Returns:
        ``(start_date, end_date)`` corresponding to the active preset.
    """
    state_key = f"{key_prefix}_date_preset"

    # Initialise on first run
    if state_key not in st.session_state:
        st.session_state[state_key] = default

    container = st.sidebar if sidebar else st

    # Render a horizontal row of buttons
    cols = container.columns(len(PRESETS))
    for col, preset_name in zip(cols, PRESETS):
        with col:
            is_active = st.session_state[state_key] == preset_name
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                preset_name,
                key=f"{key_prefix}_dp_{preset_name}",
                width="stretch",
                type=btn_type,
            ):
                st.session_state[state_key] = preset_name
                st.rerun()

    active_preset: str = st.session_state[state_key]
    return _resolve_preset(active_preset)
