"""Streamlit dashboard entry point (JobHunter pattern)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quant_tools.dashboard.views import (
    ai_analysis,
    backtest,
    factors,
    live,
    performance,
    portfolio,
    predictions,
    risk_dashboard,
    rolling,
    scanner,
)

# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Quant-Tools",
    page_icon="📊",
)

# ── Load CSS + Fonts ──────────────────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "style.css"
_css = _CSS_PATH.read_text(encoding="utf-8")
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&'
    'family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ── Global Symbol State ──────────────────────────────────────────────

from quant_tools.dashboard.core.symbol_state import (
    get_favorites,
    init_global_symbol,
    set_active_symbol,
)

init_global_symbol()

# ── Page Slug Mapping (clean URLs) ──────────────────────────────────

PAGE_SLUGS: dict[str, str] = {
    "scanner": "\U0001f4e1 Market Scanner",
    "backtest": "\U0001f4c8 Strategy Backtester",
    "portfolio": "\U0001f4bc Portfolio Analyzer",
    "factors": "\U0001f3db\ufe0f Factors & Macro",
    "predictions": "\U0001f52e Predictions",
    "performance": "\U0001f4ca Performance",
    "rolling": "\U0001f4c9 Rolling Analytics",
    "risk": "\U0001f6e1\ufe0f Risk Dashboard",
    "live": "\u26a1 Live Monitor",
    "ai": "\U0001f916 AI Analysis",
}
SLUG_REVERSE: dict[str, str] = {v: k for k, v in PAGE_SLUGS.items()}

# ── Page Registry ─────────────────────────────────────────────────────

PAGES: dict[str, object] = {
    "📡 Market Scanner": scanner,
    "📈 Strategy Backtester": backtest,
    "💼 Portfolio Analyzer": portfolio,
    "🏛️ Factors & Macro": factors,
    "🔮 Predictions": predictions,
    "📊 Performance": performance,
    "📉 Rolling Analytics": rolling,
    "🛡️ Risk Dashboard": risk_dashboard,
    "⚡ Live Monitor": live,
    "🤖 AI Analysis": ai_analysis,
}

GROUPS = {
    "DISCOVER": ["📡 Market Scanner"],
    "BUILD": ["📈 Strategy Backtester", "💼 Portfolio Analyzer"],
    "ANALYZE": ["🏛️ Factors & Macro", "🔮 Predictions", "📊 Performance"],
    "MONITOR": ["📉 Rolling Analytics", "🛡️ Risk Dashboard", "⚡ Live Monitor"],
    "TOOLS": ["🤖 AI Analysis"],
}

SCAFFOLDED = {"📉 Rolling Analytics", "🛡️ Risk Dashboard", "⚡ Live Monitor"}

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="brand-icon">📊</span>'
        '<span class="brand-title">Quant-Tools</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Watchlist (favorites) ────────────────────────────────────
    _favs = get_favorites()
    if _favs:
        st.markdown(
            '<div class="sidebar-group-label">WATCHLIST</div>',
            unsafe_allow_html=True,
        )
        for _fav_sym in _favs:
            if st.button(
                f"\u2605 {_fav_sym}",
                use_container_width=True,
                key=f"fav_{_fav_sym}",
            ):
                set_active_symbol(_fav_sym)
                st.rerun()
        st.divider()

    # Initialize active page from query params on first load
    if "active_page" not in st.session_state:
        qp = st.query_params.get("page", "📡 Market Scanner")
        # Support both slug URLs (?page=backtest) and full names (?page=📈 Strategy Backtester)
        if qp in PAGES:
            st.session_state["active_page"] = qp
        elif qp in PAGE_SLUGS:
            st.session_state["active_page"] = PAGE_SLUGS[qp]
        else:
            st.session_state["active_page"] = "📡 Market Scanner"

    # Grouped button navigation
    for group_label, group_pages in GROUPS.items():
        st.markdown(
            f'<div class="sidebar-group-label">{group_label}</div>',
            unsafe_allow_html=True,
        )
        for pg in group_pages:
            label = f"{pg} \u25E6" if pg in SCAFFOLDED else pg
            is_active = st.session_state["active_page"] == pg
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                label,
                use_container_width=True,
                type=btn_type,
                key=f"nav_{pg}",
            ):
                st.session_state["active_page"] = pg
                st.query_params["page"] = SLUG_REVERSE.get(pg, pg)
                st.rerun()

    # Sidebar footer
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">Quant-Tools Dashboard<br>'
        '<span style="font-size:0.7rem;opacity:0.6;">Data: Yahoo Finance &middot; CCXT &middot; FRED</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("? Guide", use_container_width=True, key="sidebar_guide"):
            st.session_state["show_guide"] = True
            st.rerun()
    with col2:
        if st.button("Clear Cache", use_container_width=True, key="clear_cache"):
            from quant_tools.dashboard.core.cache import clear_cache

            clear_cache()
            st.rerun()

# ── Onboarding Banner ────────────────────────────────────────────────
if not st.session_state.get("onboarding_dismissed"):
    st.markdown(
        '<div class="onboarding-banner">'
        '<div class="onboarding-title">Welcome to Quant-Tools</div>'
        '<div class="onboarding-text">A quantitative finance toolkit for scanning markets, '
        'backtesting strategies, analyzing portfolios, and tracking ML predictions.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("Start with the Scanner", type="primary", key="onboard_scanner"):
            st.session_state["onboarding_dismissed"] = True
            st.session_state["active_page"] = "📡 Market Scanner"
            st.query_params["page"] = "📡 Market Scanner"
            st.rerun()
    with cols[1]:
        if st.button("Open the Guide", key="onboard_guide"):
            st.session_state["onboarding_dismissed"] = True
            st.session_state["show_guide"] = True
            st.rerun()

# ── Command Bar ──────────────────────────────────────────────────────

from quant_tools.dashboard.core.command_bar import parse_command

_cmd_input = st.text_input(
    "",
    placeholder="Type / for commands or search symbols... (press /)",
    key="command_bar",
    label_visibility="collapsed",
)
if _cmd_input:
    _cmd_result = parse_command(_cmd_input)
    if _cmd_result.page and _cmd_result.page in PAGES:
        st.session_state["active_page"] = _cmd_result.page
        st.query_params["page"] = SLUG_REVERSE.get(_cmd_result.page, _cmd_result.page)
    if _cmd_result.symbol:
        set_active_symbol(_cmd_result.symbol)
    if _cmd_result.page or _cmd_result.symbol:
        st.rerun()

# ── Keyboard Shortcut (/ to focus search) ────────────────────────────

st.components.v1.html(
    """<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
        e.preventDefault();
        const inputs = doc.querySelectorAll('input[type="text"]');
        for (const inp of inputs) {
            if (inp.placeholder && inp.placeholder.includes('commands')) {
                inp.focus();
                break;
            }
        }
    }
});
</script>""",
    height=0,
)

# ── Page Render ───────────────────────────────────────────────────────

page_name = st.session_state.get("active_page", "📡 Market Scanner")

if st.session_state.get("show_guide"):
    from quant_tools.dashboard.views import guide

    guide.render()
elif page_name in PAGES:
    try:
        PAGES[page_name].render()  # type: ignore[union-attr]
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.exception(e)
else:
    st.error(f"Page not found: {page_name}")
