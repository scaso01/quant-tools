"""Color constants for the dashboard (JobHunter pattern)."""

from __future__ import annotations

# Accent colors
ACCENT_GREEN = "#10b981"
ACCENT_BLUE = "#3b82f6"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_ORANGE = "#f97316"

# Text colors
TEXT_PRIMARY = "#f8fafc"
TEXT_MUTED = "#64748b"
TEXT_DIM = "#475569"

# Background colors
BG_PRIMARY = "#0e1117"
BG_CARD = "#1c2128"
BG_CARD_HOVER = "#22272e"

# Border colors
BORDER_SUBTLE = "rgba(99, 102, 241, 0.1)"
BORDER_CARD = "rgba(99, 102, 241, 0.15)"

# Glow effects
GLOW_GREEN = "rgba(16, 185, 129, 0.4)"
GLOW_BLUE = "rgba(59, 130, 246, 0.4)"
GLOW_AMBER = "rgba(245, 158, 11, 0.3)"


def hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to 'r,g,b' string for rgba()."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))
