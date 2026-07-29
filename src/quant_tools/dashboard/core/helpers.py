"""Shared formatting helpers, Plotly theme, and news fetcher."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

from quant_tools.dashboard.core.cache import cached
from quant_tools.dashboard.core.colors import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_PRIMARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from quant_tools.models import MacroRegime

log = logging.getLogger("quant_tools")


# ── Formatting ────────────────────────────────────────────────────────


def format_pct(value: float | None, decimals: int = 2) -> str:
    """Format a fraction as percentage: 0.15 -> '15.00%'."""
    if value is None:
        return "N/A"
    return f"{value * 100:,.{decimals}f}%"


def format_currency(value: float | None) -> str:
    """Format a dollar value: 123456 -> '$123,456'."""
    if value is None:
        return "N/A"
    return f"${value:,.0f}"


def format_number(value: float | None, decimals: int = 2) -> str:
    """Format a number with commas: 1234.56 -> '1,234.56'."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def regime_color(regime: MacroRegime) -> str:
    """Map a macro regime to an accent color."""
    return {
        MacroRegime.EXPANSION: ACCENT_GREEN,
        MacroRegime.SLOWDOWN: ACCENT_AMBER,
        MacroRegime.CONTRACTION: ACCENT_RED,
        MacroRegime.RECOVERY: ACCENT_BLUE,
    }.get(regime, TEXT_MUTED)


def directional_color(value: float) -> str:
    """Return green for positive, red for negative, amber for near-zero values.

    Args:
        value: Numeric value to evaluate.

    Returns:
        Hex color string from the dashboard accent palette.
    """
    if value > 0.001:
        return ACCENT_GREEN
    if value < -0.001:
        return ACCENT_RED
    return ACCENT_AMBER


def directional_arrow(value: float) -> str:
    """Return an arrow character indicating direction.

    Args:
        value: Numeric value to evaluate.

    Returns:
        Up arrow, down arrow, or right arrow string.
    """
    if value > 0.001:
        return "\u25b2"
    if value < -0.001:
        return "\u25bc"
    return "\u25b6"


def format_directional(value: float, formatter: Any = None) -> str:
    """Format a value with a directional arrow prefix.

    Args:
        value: Numeric value to format.
        formatter: Callable to format the value (defaults to format_pct).

    Returns:
        String like '▲ 12.34%' or '▼ -3.21%'.
    """
    if formatter is None:
        formatter = format_pct
    arrow = directional_arrow(value)
    return f"{arrow} {formatter(value)}"


# ── Plotly Theme ──────────────────────────────────────────────────────

PLOTLY_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": BG_PRIMARY,
    "plot_bgcolor": BG_PRIMARY,
    "font": {"family": "Inter, sans-serif", "color": TEXT_PRIMARY, "size": 13},
    "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 11, "color": TEXT_MUTED},
    },
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "zerolinecolor": "rgba(255,255,255,0.1)",
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.06)",
        "zerolinecolor": "rgba(255,255,255,0.1)",
    },
}


# ── News Fetcher ──────────────────────────────────────────────────────


@cached(ttl=600, key_prefix="news")
def fetch_news(symbol: str, max_items: int = 5) -> list[dict[str, Any]]:
    """Fetch recent news for a ticker via yfinance.

    yfinance >=0.2.36 returns news as::

        [{"id": "...", "content": {"title": "...", "pubDate": "ISO-8601",
          "provider": {"displayName": "..."}, "canonicalUrl": {"url": "..."}}}]

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        max_items: Maximum news items to return.

    Returns:
        List of dicts with title, link, publisher, pubDate (ISO string).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        results = []
        for item in news[:max_items]:
            content = item.get("content", item)  # fallback to flat dict
            provider = content.get("provider", {})
            canonical = content.get("canonicalUrl", {})
            click_through = content.get("clickThroughUrl", {})
            link = (
                canonical.get("url")
                or click_through.get("url")
                or content.get("link", "#")
            )
            results.append(
                {
                    "title": content.get("title", "Untitled"),
                    "link": link,
                    "publisher": provider.get("displayName")
                    or content.get("publisher", "Unknown"),
                    "pubDate": content.get("pubDate")
                    or content.get("displayTime", ""),
                }
            )
        return results
    except Exception:
        log.warning("Failed to fetch news for %s", symbol)
        return []


def render_news_section(symbols: list[str]) -> None:
    """Render a tabbed news section for multiple symbols.

    Args:
        symbols: List of ticker symbols to show news for.
    """
    if not symbols:
        return

    st.markdown("### Latest News")

    tabs = st.tabs(symbols)
    for tab, symbol in zip(tabs, symbols):
        with tab:
            with st.spinner(f"Loading news for {symbol}..."):
                articles = fetch_news(symbol)

            if not articles:
                st.info(f"No recent news for {symbol}")
                continue

            for article in articles:
                pub_date = article.get("pubDate", "")
                time_str = "Unknown"
                if pub_date:
                    try:
                        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        time_str = dt.strftime("%b %d, %Y %I:%M %p UTC")
                    except (ValueError, TypeError):
                        time_str = pub_date

                st.markdown(
                    f'<div class="news-item">'
                    f'<a href="{article["link"]}" target="_blank" '
                    f'style="color:{ACCENT_BLUE};text-decoration:none;font-weight:600;'
                    f'font-size:0.95rem;">{article["title"]}</a>'
                    f'<div style="display:flex;gap:16px;margin-top:4px;'
                    f'font-size:0.8rem;color:{TEXT_MUTED};">'
                    f"<span>{article['publisher']}</span>"
                    f"<span>{time_str}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )


def _render_sparkline_svg(
    data: list[float], color: str, width: int = 80, height: int = 24
) -> str:
    """Generate a minimal SVG sparkline from a list of values.

    Args:
        data: Sequence of numeric values (at least 2 needed).
        color: Stroke color for the polyline.
        width: SVG viewport width in pixels.
        height: SVG viewport height in pixels.

    Returns:
        SVG markup string, or empty string if data is insufficient.
    """
    if not data or len(data) < 2:
        return ""
    min_val, max_val = min(data), max(data)
    val_range = max_val - min_val or 1.0
    points: list[str] = []
    for i, val in enumerate(data):
        x = i * width / (len(data) - 1)
        y = height - ((val - min_val) / val_range) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")
    pts = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def render_metric_card(
    label: str,
    value: str,
    color: str = ACCENT_GREEN,
    glossary_key: str = "",
    delta: float | None = None,
    delta_suffix: str = "",
    sparkline: list[float] | None = None,
    size: str = "normal",
) -> None:
    """Render a single styled metric card.

    Args:
        label: Card label text.
        value: Card value text.
        color: Value text color.
        glossary_key: If set, wraps label in an <abbr> with glossary definition.
        delta: Optional change value. Shows directional arrow and value.
        delta_suffix: Suffix for the delta value (e.g. '%' or 'bps').
        sparkline: Optional list of floats to render as a mini SVG chart.
        size: Card size variant — 'large', 'normal', or 'compact'.
    """
    if glossary_key:
        from quant_tools.dashboard.core.page_help import jargon
        # Decorate the caller's label, do not replace it. Substituting the
        # glossary key made the backtest page render two different cards, Total
        # Return and CAGR, both labelled "CAGR".
        display_label = jargon(glossary_key, label)
    else:
        display_label = label

    size_class = f" metric-card-{size}" if size != "normal" else ""

    delta_html = ""
    if delta is not None:
        arrow = directional_arrow(delta)
        d_color = directional_color(delta)
        delta_html = (
            f'<div class="metric-card-delta" style="color:{d_color};">'
            f"{arrow} {delta:+.2f}{delta_suffix}</div>"
        )

    sparkline_html = ""
    if sparkline:
        svg = _render_sparkline_svg(sparkline, color)
        if svg:
            sparkline_html = f'<div class="metric-card-sparkline">{svg}</div>'

    st.markdown(
        f'<div class="metric-card{size_class}">'
        f'<div class="metric-card-value" style="color:{color};">{value}</div>'
        f"{delta_html}"
        f'<div class="metric-card-label">{display_label}</div>'
        f"{sparkline_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Chart Builder Delegates ──────────────────────────────────────────


def build_model_agreement_chart(predictions: list[dict]):
    """Build model agreement stacked bar chart."""
    from quant_tools.dashboard.views.predictions import _build_model_agreement_chart
    return _build_model_agreement_chart(predictions)


def build_confidence_histogram(predictions: list[dict]):
    """Build confidence distribution histogram."""
    from quant_tools.dashboard.views.predictions import _build_confidence_histogram
    return _build_confidence_histogram(predictions)


def build_win_rate_chart(performance):
    """Build win rate bar chart."""
    from quant_tools.dashboard.views.performance import _build_win_rate_chart
    return _build_win_rate_chart(performance)


def build_heatmap(performance):
    """Build win rate heatmap."""
    from quant_tools.dashboard.views.performance import _build_heatmap
    return _build_heatmap(performance)
