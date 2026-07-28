"""Market Scanner view for the Streamlit dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from quant_tools.dashboard.core.colors import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    TEXT_MUTED,
)
from quant_tools.dashboard.core.helpers import (
    PLOTLY_LAYOUT,
    directional_color,
    format_number,
    render_metric_card,
    render_news_section,
)
from quant_tools.data.universe import (
    SECTOR_MAP,
    UNIVERSE_PRESETS,
    UniversePreset,
    COMMODITY_PRESETS,
    COMMODITY_UNIVERSE,
    list_all_presets,
    _get_extended_preset,
)
from quant_tools.dashboard.core.page_help import render_page_help
from quant_tools.errors import ComputeError, DataError, NetworkError
from quant_tools.models import AssetClass, ScanResult, SignalType
from quant_tools.scanner.engine import scan_universe
from quant_tools.scanner.signal_explanations import (
    SIGNAL_EXPLANATIONS,
    SignalExplanation,
    SignalComponent,
)


# ── Cached scan wrapper ──────────────────────────────────────────────


_ASSET_CLASS_MAP: dict[str, AssetClass] = {
    "Equity": AssetClass.EQUITY,
    "Crypto": AssetClass.CRYPTO,
    "Forex": AssetClass.FOREX,
    "Commodity": AssetClass.COMMODITY,
}

# Timeframe presets: label → (period, interval)
_TIMEFRAME_PRESETS: dict[str, tuple[str, str]] = {
    # Intraday — bar size is the value, lookback auto-set
    "Intraday · 1 min": ("1d", "1m"),
    "Intraday · 5 min": ("5d", "5m"),
    "Intraday · 15 min": ("5d", "15m"),
    "Intraday · 30 min": ("5d", "30m"),
    "Intraday · 1 hour": ("1mo", "1h"),
    "Intraday · 4 hours": ("3mo", "4h"),
    "Intraday · 6 hours": ("6mo", "6h"),
    "Intraday · 8 hours": ("6mo", "8h"),
    "Intraday · 12 hours": ("1y", "12h"),
    # Daily — 1d bars, lookback varies
    "Daily · 1 day": ("1d", "1d"),
    "Daily · 5 days": ("5d", "1d"),
    "Daily · 1 week": ("5d", "1d"),
    # Monthly — 1d bars, month-scale lookback
    "Monthly · 1 month": ("1mo", "1d"),
    "Monthly · 3 months": ("3mo", "1d"),
    "Monthly · 6 months": ("6mo", "1d"),
    "Monthly · 12 months": ("1y", "1d"),
    # Years — weekly/monthly bars, long lookback
    "Years · 1 year": ("1y", "1wk"),
    "Years · 2 years": ("2y", "1wk"),
    "Years · 5 years": ("5y", "1wk"),
    "Years · 10 years": ("10y", "1mo"),
    "Years · Max": ("max", "1mo"),
}

# Asset class support per timeframe (multi-hour intervals are CCXT-only)
_ALL_CLASSES = frozenset({"Equity", "Crypto", "Forex", "Commodity"})
_CRYPTO_ONLY = frozenset({"Crypto"})

_TIMEFRAME_SUPPORT: dict[str, frozenset[str]] = {k: _ALL_CLASSES for k in _TIMEFRAME_PRESETS}
# Override crypto-only entries (yfinance has no 4h/6h/8h/12h intervals)
for _label in ("Intraday · 4 hours", "Intraday · 6 hours",
               "Intraday · 8 hours", "Intraday · 12 hours"):
    _TIMEFRAME_SUPPORT[_label] = _CRYPTO_ONLY

_DEFAULT_TIMEFRAME = "Monthly · 3 months"


def _get_timeframes(asset_class: str) -> list[str]:
    """Return timeframe labels available for the given asset class."""
    return [
        label for label, classes in _TIMEFRAME_SUPPORT.items()
        if asset_class in classes
    ]


@st.cache_data(ttl=60, show_spinner=False)
def _cached_scan(
    symbols_key: str,
    signals_key: str,
    top_n: int,
    period: str,
    asset_class_key: str = "Equity",
    interval: str = "1d",
) -> list[ScanResult]:
    """Cached wrapper around scan_universe.

    Uses string keys for hashability (st.cache_data requirement).
    Results cached 60s — short enough for near-realtime feel with
    auto-refresh, long enough to avoid hammering yfinance.
    """
    symbols = symbols_key.split(",")
    signal_map = {s.value: s for s in SignalType}
    signals = [signal_map[s] for s in signals_key.split(",")]
    asset_class = _ASSET_CLASS_MAP.get(asset_class_key, AssetClass.EQUITY)
    return scan_universe(
        symbols=symbols,
        signals=signals,
        top_n=top_n,
        period=period,
        asset_class=asset_class,
        interval=interval,
    )


# ── Signal display config ────────────────────────────────────────────

_SIGNAL_COLORS: dict[SignalType, str] = {
    SignalType.MOMENTUM: ACCENT_BLUE,
    SignalType.VALUE: ACCENT_GREEN,
    SignalType.QUALITY: ACCENT_PURPLE,
    SignalType.MEAN_REVERSION: ACCENT_AMBER,
}

_SIGNAL_LABELS: dict[SignalType, str] = {
    SignalType.MOMENTUM: "Momentum",
    SignalType.VALUE: "Value",
    SignalType.QUALITY: "Quality",
    SignalType.MEAN_REVERSION: "Mean Reversion",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _results_to_dataframe(results: list[ScanResult]) -> pd.DataFrame:
    """Convert scan results to a styled DataFrame for display."""
    rows = []
    for r in results:
        rows.append(
            {
                "Symbol": r.symbol,
                "Name": r.name,
                "Sector": r.sector,
                "Signal": _SIGNAL_LABELS.get(r.signal, r.signal.value),
                "Score": r.score,
                "Price": r.price,
                "1D%": r.change_1d,
                "5D%": r.change_5d,
                "1M%": r.change_1m,
                "RSI": r.rsi_14 if r.rsi_14 is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _build_score_chart(results: list[ScanResult]) -> None:
    """Render a score distribution bar chart."""
    df = pd.DataFrame(
        {
            "Symbol": [r.symbol for r in results],
            "Score": [r.score for r in results],
            "Signal": [_SIGNAL_LABELS.get(r.signal, r.signal.value) for r in results],
        }
    )

    color_map = {
        _SIGNAL_LABELS[k]: v for k, v in _SIGNAL_COLORS.items()
    }

    fig = px.bar(
        df,
        x="Symbol",
        y="Score",
        color="Signal",
        color_discrete_map=color_map,
        title="Score Distribution",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _build_signal_donut(results: list[ScanResult]) -> None:
    """Render a signal breakdown donut chart."""
    counts: Counter[str] = Counter()
    for r in results:
        counts[_SIGNAL_LABELS.get(r.signal, r.signal.value)] += 1

    df = pd.DataFrame(
        {"Signal": list(counts.keys()), "Count": list(counts.values())}
    )

    color_map = {
        _SIGNAL_LABELS[k]: v for k, v in _SIGNAL_COLORS.items()
    }

    fig = px.pie(
        df,
        names="Signal",
        values="Count",
        hole=0.4,
        title="Signal Breakdown",
        color="Signal",
        color_discrete_map=color_map,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _build_treemap(results: list[ScanResult]) -> None:
    """Render a sector/symbol treemap colored by daily change.

    Uses ``plotly.express.treemap`` with ``path=["sector", "symbol"]``.
    Color encodes 1-day percentage change on a diverging red-green scale.
    Clicking a symbol sets session state so other pages can pick it up.
    """
    if not results:
        st.info("Run a scan to see the market map.")
        return

    rows = []
    for r in results:
        # Use absolute score as the tile size (ensures positive values for treemap)
        size_val = max(abs(r.score), 0.01)
        rows.append(
            {
                "sector": r.sector or "Unknown",
                "symbol": r.symbol,
                "score": r.score,
                "change_1d": r.change_1d,
                "size": size_val,
                "label": f"{r.symbol}<br>{r.change_1d:+.2f}%",
            }
        )

    df = pd.DataFrame(rows)

    fig = px.treemap(
        df,
        path=["sector", "symbol"],
        values="size",
        color="change_1d",
        color_continuous_scale=[
            [0.0, ACCENT_RED],       # #ef4444 (red for negative)
            [0.5, "#1c2128"],        # dark neutral midpoint
            [1.0, ACCENT_GREEN],     # #10b981 (green for positive)
        ],
        color_continuous_midpoint=0.0,
        custom_data=["symbol", "score", "change_1d"],
        title="Market Map (size = score, color = 1D change)",
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[2]:+.2f}%",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Score: %{customdata[1]:.2f}<br>"
            "1D Change: %{customdata[2]:+.2f}%"
            "<extra>%{parent}</extra>"
        ),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_colorbar={"title": "1D %", "ticksuffix": "%"},
    )

    # Render the chart and capture click events
    clicked = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                              key="scanner_treemap_click")

    # Handle click: navigate to backtester with clicked symbol
    if clicked and clicked.selection and clicked.selection.points:
        point = clicked.selection.points[0]
        # Extract symbol from customdata or label
        click_symbol = ""
        if hasattr(point, "customdata") and point.customdata:
            click_symbol = str(point.customdata[0])
        elif hasattr(point, "label"):
            click_symbol = str(point.label).split("<br>")[0].strip()

        if click_symbol and click_symbol not in ("Unknown", ""):
            st.session_state["backtest_prefill_ticker"] = click_symbol
            st.session_state["active_page"] = "\U0001f4c8 Strategy Backtester"
            st.query_params["page"] = "\U0001f4c8 Strategy Backtester"
            st.rerun()


# ── Main render ──────────────────────────────────────────────────────


def render() -> None:
    """Render the Market Scanner page."""

    # ── Apply recommended settings if requested ─────────────────────
    if st.session_state.pop("_scanner_apply_defaults", False):
        st.session_state["scanner_asset_class"] = "Equity"
        st.session_state["scanner_universe"] = "S&P 500"
        st.session_state["scanner_sig_momentum"] = True
        st.session_state["scanner_sig_value"] = True
        st.session_state["scanner_sig_quality"] = True
        st.session_state["scanner_sig_meanrev"] = True
        st.session_state["scanner_top_n"] = 20

    # ── Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-header">Scanner Settings</div>',
            unsafe_allow_html=True,
        )

        # ── Asset Class ────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-group-label">Asset Class</div>',
            unsafe_allow_html=True,
        )

        asset_class_option = st.selectbox(
            "Asset Class",
            ["Equity", "Crypto", "Forex", "Commodity"],
            index=0,
            key="scanner_asset_class",
            label_visibility="collapsed",
            help="Choose which type of financial instrument to scan",
        )

        # ── Universe ────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-group-label">Universe</div>',
            unsafe_allow_html=True,
        )

        # Build combined preset names (static + extended)
        _EXTENDED_NAMES = [
            "S&P 500", "Russell 1000", "International EAFE",
            "Emerging Markets", "Global All",
        ]
        universe_names = list(UNIVERSE_PRESETS.keys()) + _EXTENDED_NAMES
        selected_universe_name = st.selectbox(
            "Universe",
            universe_names,
            index=0,
            key="scanner_universe",
            label_visibility="collapsed",
            disabled=(asset_class_option != "Equity"),
            help="The pool of stocks to scan — larger universes take longer",
        )

        # Resolve the preset: static first, then extended
        if selected_universe_name in UNIVERSE_PRESETS:
            preset: UniversePreset = UNIVERSE_PRESETS[selected_universe_name]
        else:
            _ext = _get_extended_preset(selected_universe_name)
            if _ext is not None:
                preset = _ext
            else:
                # Fallback to first static preset if extended load fails
                preset = next(iter(UNIVERSE_PRESETS.values()))

        # Large universe warning
        if preset.count > 200:
            st.warning(
                f"Large universe ({preset.count} stocks) "
                "\u2014 scan may take 30\u201360 seconds."
            )

        # Dynamic info badge
        st.markdown(
            f'<div class="universe-info">'
            f'<span class="universe-dot"></span>'
            f"{preset.description} ({preset.count} stocks)"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Commodity Category (only when Commodity asset class) ──
        commodity_category: str | None = None
        if asset_class_option == "Commodity":
            st.markdown(
                '<div class="sidebar-group-label">Commodity Category</div>',
                unsafe_allow_html=True,
            )
            commodity_category = st.selectbox(
                "Commodity Category",
                list(COMMODITY_PRESETS.keys()),
                key="scanner_commodity_cat",
                label_visibility="collapsed",
                help="Filter commodities by category (energy, metals, etc.)",
            )

        # ── Signals (2x2 checkbox grid) ─────────────────────────
        st.markdown(
            '<div class="sidebar-group-label">Signals</div>',
            unsafe_allow_html=True,
        )

        sig_row1_l, sig_row1_r = st.columns(2)
        sig_row2_l, sig_row2_r = st.columns(2)
        with sig_row1_l:
            sig_momentum = st.checkbox(
                "Momentum", value=True, key="scanner_sig_momentum",
                help="Scan for trend strength and acceleration signals",
            )
        with sig_row1_r:
            sig_value = st.checkbox(
                "Value", value=True, key="scanner_sig_value",
                help="Scan for undervalued stocks (low P/E, high yield)",
            )
        with sig_row2_l:
            sig_quality = st.checkbox(
                "Quality", value=True, key="scanner_sig_quality",
                help="Scan for high-quality companies (strong ROE, margins)",
            )
        with sig_row2_r:
            sig_mean_rev = st.checkbox(
                "Mean Rev.", value=True, key="scanner_sig_meanrev",
                help="Scan for oversold stocks likely to bounce back",
            )

        # Build selected signals list from checkboxes
        _sig_checks = {
            SignalType.MOMENTUM: sig_momentum,
            SignalType.VALUE: sig_value,
            SignalType.QUALITY: sig_quality,
            SignalType.MEAN_REVERSION: sig_mean_rev,
        }
        selected_signals = [s for s, on in _sig_checks.items() if on]

        # ── Sector Filter (expander, only for sector-aware universes) ─
        if preset.has_sectors:
            all_sectors = sorted(SECTOR_MAP.keys())

            # Count how many are selected for the summary
            _sector_count = sum(
                1 for s in all_sectors
                if st.session_state.get(f"scanner_sec_{s}", True)
            )

            with st.expander(
                f"Sector Filter ({_sector_count}/{len(all_sectors)})",
                expanded=False,
            ):
                sec_btn_l, sec_btn_r = st.columns(2)
                with sec_btn_l:
                    if st.button("All", key="scanner_sec_all",
                                 use_container_width=True):
                        for s in all_sectors:
                            st.session_state[f"scanner_sec_{s}"] = True
                        st.rerun()
                with sec_btn_r:
                    if st.button("None", key="scanner_sec_none",
                                 use_container_width=True):
                        for s in all_sectors:
                            st.session_state[f"scanner_sec_{s}"] = False
                        st.rerun()

                # 2-column checkbox grid for sectors
                sec_left, sec_right = st.columns(2)
                selected_sectors: list[str] = []
                for i, sector in enumerate(all_sectors):
                    col = sec_left if i % 2 == 0 else sec_right
                    with col:
                        checked = st.checkbox(
                            sector,
                            value=True,
                            key=f"scanner_sec_{sector}",
                        )
                        if checked:
                            selected_sectors.append(sector)
        else:
            selected_sectors = []

        # ── Parameters ───────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-group-label">Parameters</div>',
            unsafe_allow_html=True,
        )

        top_n = st.slider(
            "Top N",
            min_value=5,
            max_value=50,
            value=20,
            key="scanner_top_n",
            help="How many top results to show, ranked by signal strength",
        )

        st.markdown(
            '<div class="sidebar-group-label">Timeframe</div>',
            unsafe_allow_html=True,
        )

        timeframe_labels = _get_timeframes(asset_class_option)
        default_idx = (
            timeframe_labels.index(_DEFAULT_TIMEFRAME)
            if _DEFAULT_TIMEFRAME in timeframe_labels
            else 0
        )
        selected_timeframe = st.selectbox(
            "Timeframe",
            timeframe_labels,
            index=default_idx,
            key="scanner_timeframe",
            label_visibility="collapsed",
            help="Historical period to analyze for signals",
        )
        period, interval = _TIMEFRAME_PRESETS[selected_timeframe]

        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

        run_scan = st.button(
            "Run Scan",
            use_container_width=True,
            type="primary",
            key="scanner_run",
        )

        # ── Last scan timestamp ──────────────────────────────────
        last_run = st.session_state.get("scanner_last_run")
        if last_run:
            st.markdown(
                f'<div class="sidebar-footer">'
                f'<span>Last scan: {last_run.strftime("%#I:%M:%S %p")}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Header (dynamic subtitle based on universe) ────────────────
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>Market Scanner</h1>"
        f"<p>Scanning {preset.name} for trading signals</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    render_page_help("scanner")

    # ── Early exit if no scan triggered ──────────────────────────────
    if not run_scan and "scanner_results" not in st.session_state:
        st.info(
            "Ready to scan! Try these recommended settings: **S&P 500** universe, "
            "**all signals** selected, **Top 20** results."
        )
        if st.button("Use recommended settings", key="scanner_quick_start"):
            st.session_state["_scanner_apply_defaults"] = True
            st.rerun()
        return

    # ── Build symbol list from selections ────────────────────────────
    if run_scan:
        # Non-equity asset classes use their own symbol lists
        _CRYPTO_SYMBOLS = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "AVAX/USDT", "DOT/USDT", "DOGE/USDT",
            "LINK/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
            "APT/USDT", "ALGO/USDT", "TRX/USDT",
        ]
        _FOREX_SYMBOLS = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
            "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
        ]
        if asset_class_option == "Crypto":
            unique_symbols = _CRYPTO_SYMBOLS
        elif asset_class_option == "Forex":
            unique_symbols = _FOREX_SYMBOLS
        elif asset_class_option == "Commodity":
            # Use selected commodity category, fall back to all commodities
            _cat = commodity_category or "All Commodities"
            unique_symbols = list(COMMODITY_PRESETS.get(_cat, COMMODITY_PRESETS["All Commodities"]))
        elif preset.has_sectors and selected_sectors:
            # Filter universe by selected sectors
            sector_tickers: set[str] = set()
            for sector in selected_sectors:
                sector_tickers.update(SECTOR_MAP.get(sector, []))
            unique_symbols = [
                s for s in preset.symbols if s in sector_tickers
            ]
        elif preset.has_sectors and not selected_sectors:
            unique_symbols = []
        else:
            # Non-sector universes use all symbols directly
            unique_symbols = list(preset.symbols)

        if not unique_symbols:
            st.warning("No symbols match the current selections.")
            return

        if not selected_signals:
            st.warning("Select at least one signal type.")
            return

        # ── Run the scan ─────────────────────────────────────────
        try:
            symbols_key = ",".join(unique_symbols)
            signals_key = ",".join(s.value for s in selected_signals)
            st.session_state["_last_symbols_key"] = symbols_key
            st.session_state["_last_signals_key"] = signals_key
            progress = st.progress(0, text="Scanning universe...")
            progress.progress(10, text=f"Scanning {len(unique_symbols)} symbols...")
            results = _cached_scan(symbols_key, signals_key, top_n, period, asset_class_option, interval)
            progress.progress(100, text="Scan complete!")
            st.session_state["scanner_results"] = results
            st.session_state["scanner_count"] = len(unique_symbols)
            st.session_state["scanner_last_run"] = datetime.now()
        except DataError as exc:
            st.warning(f"Data issue: {exc}")
            return
        except NetworkError as exc:
            st.error(f"Network error: {exc}")
            return
        except ComputeError as exc:
            st.warning(f"Computation error: {exc}")
            return

    # ── Retrieve cached results ──────────────────────────────────────
    results: list[ScanResult] = st.session_state.get("scanner_results", [])
    scanned_count: int = st.session_state.get("scanner_count", 0)

    if not results:
        st.warning("Scan returned no results. Try broadening your filters.")
        return

    # ── Auto-refresh (inline, not sidebar) ────────────────────────
    with st.expander("Auto-Refresh Settings", expanded=False):
        ar_left, ar_right = st.columns([1, 2])
        with ar_left:
            auto_refresh = st.toggle(
                "Enable",
                value=False,
                key="scanner_auto_refresh",
                help="Automatically re-scan at the selected interval",
            )
        with ar_right:
            refresh_interval = st.select_slider(
                "Interval",
                options=[60, 90, 120, 180, 300],
                value=60,
                format_func=lambda x: (
                    f"{x // 60}m {x % 60}s".replace(" 0s", "")
                    if x >= 60
                    else f"{x}s"
                ),
                key="scanner_refresh_interval",
                disabled=not auto_refresh,
            )

    # ── Store params for auto-refresh ─────────────────────────────
    st.session_state["_scan_params"] = {
        "symbols_key": st.session_state.get("_last_symbols_key", ""),
        "signals_key": st.session_state.get("_last_signals_key", ""),
        "top_n": top_n,
        "period": period,
        "asset_class_key": asset_class_option,
        "interval": interval,
    }

    # ── Auto-refresh fragment ─────────────────────────────────────
    _render_scanner_results(results, scanned_count, auto_refresh, refresh_interval)


def _render_scanner_results(
    initial_results: list[ScanResult],
    scanned_count: int,
    auto_refresh: bool,
    refresh_interval: int,
) -> None:
    """Render scanner results, optionally with auto-refresh."""

    if auto_refresh:
        @st.fragment(run_every=timedelta(seconds=refresh_interval))
        def _auto_refresh_content() -> None:
            _render_results_content(scanned_count)

        _auto_refresh_content()
    else:
        _render_results_content(scanned_count)


def _render_results_content(scanned_count: int) -> None:
    """Render KPIs, table, charts, and news from session_state results.

    When called inside an auto-refresh fragment, this re-runs the scan
    (cache handles dedup) and updates the display.
    """
    # Re-fetch if auto-refresh is active (cache returns instantly if fresh)
    params = st.session_state.get("_scan_params", {})
    if params.get("symbols_key"):
        try:
            results = _cached_scan(
                params["symbols_key"],
                params["signals_key"],
                params["top_n"],
                params["period"],
                params.get("asset_class_key", "Equity"),
                params.get("interval", "1d"),
            )
            st.session_state["scanner_results"] = results
        except (DataError, NetworkError, ComputeError):
            pass  # Keep previous results on refresh failure

    results: list[ScanResult] = st.session_state.get("scanner_results", [])
    if not results:
        return

    # ── KPI Row ──────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        render_metric_card("Stocks Scanned", str(scanned_count), ACCENT_BLUE)

    with kpi2:
        avg_score = sum(r.score for r in results) / len(results)
        render_metric_card(
            "Avg Score",
            format_number(avg_score),
            ACCENT_GREEN,
            glossary_key="Signal Score",
        )

    with kpi3:
        signal_counts: Counter[SignalType] = Counter(r.signal for r in results)
        top_signal = signal_counts.most_common(1)[0][0]
        render_metric_card(
            "Top Signal",
            _SIGNAL_LABELS.get(top_signal, top_signal.value),
            _SIGNAL_COLORS.get(top_signal, ACCENT_BLUE),
        )

    with kpi4:
        best = results[0]
        render_metric_card(
            "Best Performer",
            f"{best.symbol} ({format_number(best.score)})",
            ACCENT_GREEN,
            delta=best.change_1d,
            delta_suffix="%",
        )

    st.markdown("")

    # ── Results Table ────────────────────────────────────────────────
    st.markdown("### Scan Results")

    df = _results_to_dataframe(results)

    st.dataframe(
        df.style.map(
            lambda v: f"color: {ACCENT_GREEN}" if isinstance(v, (int, float)) and v > 0
            else f"color: {ACCENT_RED}" if isinstance(v, (int, float)) and v < 0
            else "",
            subset=["1D%", "5D%", "1M%"],
        ).format(
            {
                "Score": "{:.2f}",
                "Price": "${:.2f}",
                "1D%": "{:+.2f}%",
                "5D%": "{:+.2f}%",
                "1M%": "{:+.2f}%",
                "RSI": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=min(400, 40 + 35 * len(results)),
    )

    st.markdown("")

    # ── Signal Legend ─────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;gap:24px;flex-wrap:wrap;'
        'padding:4px 0 12px 0;font-size:0.8rem;line-height:1.4;">'
        f'<span><span style="color:{ACCENT_BLUE};font-weight:600;">'
        f"Momentum</span>"
        f' <span style="color:{TEXT_MUTED};">\u2014 trend strength &amp; acceleration</span></span>'
        f'<span><span style="color:{ACCENT_GREEN};font-weight:600;">'
        f"Value</span>"
        f' <span style="color:{TEXT_MUTED};">\u2014 low P/E, P/B, high yield</span></span>'
        f'<span><span style="color:{ACCENT_PURPLE};font-weight:600;">'
        f"Quality</span>"
        f' <span style="color:{TEXT_MUTED};">\u2014 high ROE, margins, low debt</span></span>'
        f'<span><span style="color:{ACCENT_AMBER};font-weight:600;">'
        f"Mean Reversion</span>"
        f' <span style="color:{TEXT_MUTED};">\u2014 oversold below moving avg</span></span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Next Steps ────────────────────────────────────────────────────
    if results:
        st.markdown("---")
        next_cols = st.columns([1, 1, 4])
        with next_cols[0]:
            if st.button("Backtest top result \u2192", key="scanner_to_backtest"):
                top_symbol = results[0].symbol
                st.session_state["backtest_prefill_ticker"] = top_symbol
                st.session_state["active_page"] = "\U0001f4c8 Strategy Backtester"
                st.query_params["page"] = "\U0001f4c8 Strategy Backtester"
                st.rerun()
        with next_cols[1]:
            if st.button("Get Predictions \u2192", key="scanner_to_predictions"):
                top_symbol = results[0].symbol
                st.session_state["predictions_prefill_symbol"] = top_symbol
                st.session_state["active_page"] = "\U0001f52e Predictions"
                st.query_params["page"] = "\U0001f52e Predictions"
                st.rerun()

    # ── Per-Result Score Breakdowns ──────────────────────────────────
    results_with_breakdown = [r for r in results if r.details.get("breakdown")]
    if results_with_breakdown:
        with st.expander("Score Breakdowns", expanded=False):
            for r in results_with_breakdown[:10]:
                breakdown = r.details.get("breakdown", {})
                if breakdown:
                    st.markdown(
                        f"**{r.symbol}** \u2014 {r.signal.value} "
                        f"(total: {r.score:.1f})"
                    )
                    # breakdown is {signal_name: {component: score}}
                    # Flatten into a list of rows for display
                    breakdown_data = []
                    for signal_name, components in breakdown.items():
                        if isinstance(components, dict):
                            for comp_name, comp_score in components.items():
                                breakdown_data.append(
                                    {"Signal": signal_name, "Component": comp_name, "Score": f"{comp_score:.1f}"}
                                )
                        else:
                            # Fallback: components is already a scalar
                            breakdown_data.append(
                                {"Signal": signal_name, "Component": signal_name, "Score": f"{components:.1f}"}
                            )
                    if breakdown_data:
                        st.dataframe(
                            breakdown_data,
                            use_container_width=True,
                            hide_index=True,
                        )

    # ── Signal Explanation Panel ──────────────────────────────────────
    _asset_class_label = params.get("asset_class_key", "Equity")
    with st.expander("How Signals Are Calculated", expanded=False):
        signal_tabs = st.tabs(
            [exp.name for exp in SIGNAL_EXPLANATIONS.values()]
        )
        for tab, (sig_type, explanation) in zip(
            signal_tabs, SIGNAL_EXPLANATIONS.items()
        ):
            with tab:
                st.markdown(f"**{explanation.summary}**")
                comp_data = []
                for comp in explanation.components:
                    comp_data.append(
                        {
                            "Component": comp.name,
                            "Weight": f"{comp.weight:.0%}",
                            "Description": comp.description,
                            "Interpretation": comp.interpretation,
                        }
                    )
                st.dataframe(
                    comp_data, use_container_width=True, hide_index=True,
                )
                st.markdown("**Score Ranges:**")
                for range_str, meaning in explanation.score_interpretation.items():
                    st.markdown(f"- **{range_str}**: {meaning}")
                asset_key = _asset_class_label.lower()
                if asset_key in explanation.asset_class_notes:
                    st.info(
                        f"**{_asset_class_label} note:** "
                        f"{explanation.asset_class_notes[asset_key]}"
                    )

    # ── Charts Row (tabbed) ──────────────────────────────────────────
    chart_tabs = st.tabs(["Score Distribution", "Signal Breakdown", "Market Map"])

    with chart_tabs[0]:
        _build_score_chart(results)

    with chart_tabs[1]:
        _build_signal_donut(results)

    with chart_tabs[2]:
        _build_treemap(results)

    st.markdown("")

    # ── News Section ─────────────────────────────────────────────────
    news_symbols = [r.symbol for r in results]
    render_news_section(news_symbols)
