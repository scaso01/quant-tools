"""Formatting helpers for TUI display."""

from __future__ import annotations


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string.

    Args:
        value: Decimal value (e.g., 0.15 for 15%).
        decimals: Number of decimal places.

    Returns:
        Formatted string like "15.00%".
    """
    return f"{value * 100:.{decimals}f}%"


def fmt_currency(value: float, symbol: str = "$") -> str:
    """Format a number as currency.

    Args:
        value: Dollar amount.
        symbol: Currency symbol.

    Returns:
        Formatted string like "$1,234.56".
    """
    if abs(value) >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value:,.2f}"
    return f"{symbol}{value:.2f}"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas.

    Args:
        value: Numeric value.
        decimals: Decimal places.

    Returns:
        Formatted string like "1,234.56".
    """
    return f"{value:,.{decimals}f}"


def color_by_value(value: float, threshold: float = 0.0) -> str:
    """Return Rich color tag based on value vs threshold.

    Args:
        value: The value to color.
        threshold: The neutral threshold.

    Returns:
        Rich color string ("green", "red", or "yellow").
    """
    if value > threshold:
        return "green"
    if value < threshold:
        return "red"
    return "yellow"


def fmt_score(score: float) -> str:
    """Format a score with color.

    Args:
        score: Score value (0-100).

    Returns:
        Rich-formatted score string.
    """
    if score >= 70:
        color = "green"
    elif score >= 40:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{score:.1f}[/{color}]"


def fmt_change(value: float) -> str:
    """Format a price change with color and sign.

    Args:
        value: Change as decimal (e.g., 0.05 for +5%).

    Returns:
        Rich-formatted change string.
    """
    color = color_by_value(value)
    sign = "+" if value > 0 else ""
    return f"[{color}]{sign}{value * 100:.2f}%[/{color}]"


def bar_chart(value: float, max_value: float = 100.0, width: int = 20) -> str:
    """Create a simple text bar chart.

    Args:
        value: Current value.
        max_value: Maximum value for full bar.
        width: Width of bar in characters.

    Returns:
        Bar string like "########............".
    """
    ratio = min(max(value / max_value, 0), 1.0) if max_value > 0 else 0
    filled = int(ratio * width)
    return "\u2588" * filled + "\u2591" * (width - filled)
