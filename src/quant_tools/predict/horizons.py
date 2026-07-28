"""Asset-class-aware prediction horizon presets.

Provides per-asset-class horizon options and helpers for the prediction
pipeline, CLI, and dashboard.
"""

from __future__ import annotations

from typing import NamedTuple

from quant_tools.models import AssetClass


class HorizonPreset(NamedTuple):
    """Single horizon option."""

    value: int          # numeric value (hours or days)
    unit: str           # "h" or "d"
    label: str          # display string, e.g. "4h" or "5d"
    interval: str       # data interval: "1h", "4h", or "1d"


# Per-asset-class horizon presets
_PRESETS: dict[AssetClass, list[HorizonPreset]] = {
    AssetClass.CRYPTO: [
        HorizonPreset(1,  "h", "1h",  "1h"),
        HorizonPreset(4,  "h", "4h",  "1h"),
        HorizonPreset(8,  "h", "8h",  "4h"),
        HorizonPreset(24, "h", "24h", "4h"),
        HorizonPreset(1,  "d", "1d",  "1d"),
        HorizonPreset(5,  "d", "5d",  "1d"),
        HorizonPreset(20, "d", "20d", "1d"),
        HorizonPreset(60, "d", "60d", "1d"),
    ],
    AssetClass.EQUITY: [
        HorizonPreset(1,  "d", "1d",  "1d"),
        HorizonPreset(5,  "d", "5d",  "1d"),
        HorizonPreset(20, "d", "20d", "1d"),
        HorizonPreset(60, "d", "60d", "1d"),
    ],
    AssetClass.FOREX: [
        HorizonPreset(1,  "h", "1h",  "1h"),
        HorizonPreset(4,  "h", "4h",  "1h"),
        HorizonPreset(8,  "h", "8h",  "4h"),
        HorizonPreset(1,  "d", "1d",  "1d"),
        HorizonPreset(5,  "d", "5d",  "1d"),
        HorizonPreset(20, "d", "20d", "1d"),
    ],
    AssetClass.COMMODITY: [
        HorizonPreset(1,  "d", "1d",  "1d"),
        HorizonPreset(5,  "d", "5d",  "1d"),
        HorizonPreset(20, "d", "20d", "1d"),
        HorizonPreset(60, "d", "60d", "1d"),
    ],
}


def get_horizons(asset_class: AssetClass) -> list[HorizonPreset]:
    """Return available horizon presets for an asset class."""
    return _PRESETS.get(asset_class, _PRESETS[AssetClass.EQUITY])


def get_horizon_labels(asset_class: AssetClass) -> list[str]:
    """Return display labels for an asset class's horizons."""
    return [p.label for p in get_horizons(asset_class)]


def horizon_display(
    horizon_days: int | None = None,
    horizon_hours: int | None = None,
) -> str:
    """Format a horizon for display: '5d' or '4h'."""
    if horizon_hours is not None:
        return f"{horizon_hours}h"
    if horizon_days is not None:
        return f"{horizon_days}d"
    return "?"


def interval_for_hours(hours: int) -> str:
    """Choose data interval for an hourly horizon."""
    return "1h" if hours <= 4 else "4h"


def parse_horizon_label(label: str) -> HorizonPreset | None:
    """Parse a label like '4h' or '5d' into a HorizonPreset."""
    if not label or len(label) < 2:
        return None
    unit = label[-1]
    try:
        value = int(label[:-1])
    except ValueError:
        return None
    if unit == "h":
        interval = "1h" if value <= 4 else "4h"
    elif unit == "d":
        interval = "1d"
    else:
        return None
    return HorizonPreset(value, unit, label, interval)
