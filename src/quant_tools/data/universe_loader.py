"""Lazy-loading universe loader with caching."""
from __future__ import annotations

import json
from pathlib import Path

from quant_tools.models import TickerMeta

_UNIVERSES_DIR = Path(__file__).parent / "universes"
_cache: dict[str, dict[str, TickerMeta]] = {}


def load_universe(name: str) -> dict[str, TickerMeta]:
    """Load a universe by name (e.g., 'sp500', 'russell1000', 'international').

    Cached after first load.
    """
    if name in _cache:
        return _cache[name]
    path = _UNIVERSES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Universe '{name}' not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    universe = {
        symbol: TickerMeta(
            name=meta["name"],
            sector=meta.get("sector", ""),
            sub_industry=meta.get("sub_industry", ""),
            region=meta.get("region", "US"),
            index_membership=meta.get("index_membership", []),
        )
        for symbol, meta in raw.items()
    }
    _cache[name] = universe
    return universe


def load_combined(names: list[str]) -> dict[str, TickerMeta]:
    """Load and merge multiple universes with dedup (later universes override)."""
    combined: dict[str, TickerMeta] = {}
    for name in names:
        combined.update(load_universe(name))
    return combined


def get_ticker_metadata(symbol: str) -> TickerMeta | None:
    """Search all loaded universes for a ticker's metadata."""
    for universe in _cache.values():
        if symbol in universe:
            return universe[symbol]
    # Try loading all available universes
    for json_file in _UNIVERSES_DIR.glob("*.json"):
        name = json_file.stem
        if name not in _cache:
            load_universe(name)
            if symbol in _cache[name]:
                return _cache[name][symbol]
    return None


def clear_cache() -> None:
    """Clear the universe cache (useful for testing)."""
    _cache.clear()


def available_universes() -> list[str]:
    """Return names of all available universe JSON files."""
    return sorted(p.stem for p in _UNIVERSES_DIR.glob("*.json"))
