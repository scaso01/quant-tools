"""Symbol search index for the dashboard autocomplete picker.

Builds a flat search index from all universe sources (equities, crypto,
forex, commodities) and provides prefix/substring matching with S&P 500
prioritization.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchEntry:
    """A single searchable symbol entry."""

    symbol: str
    name: str
    sector: str
    asset_class: str


@functools.lru_cache(maxsize=1)
def build_search_index() -> tuple[SearchEntry, ...]:
    """Build and cache the full search index across all asset classes.

    Returns:
        Tuple of :class:`SearchEntry` (immutable for caching).
    """
    from quant_tools.data.universe_loader import load_combined
    from quant_tools.data.universe import (
        CRYPTO_UNIVERSE,
        FOREX_UNIVERSE,
        COMMODITY_UNIVERSE,
    )

    entries: list[SearchEntry] = []

    # ── Equities (S&P 500, Russell 1000, International) ──────────
    combined = load_combined(["sp500", "russell1000", "international"])
    for symbol, meta in combined.items():
        entries.append(
            SearchEntry(
                symbol=symbol,
                name=meta.name,
                sector=meta.sector,
                asset_class="equity",
            )
        )

    # ── Crypto ───────────────────────────────────────────────────
    for symbol, display_name in CRYPTO_UNIVERSE.items():
        entries.append(
            SearchEntry(
                symbol=symbol,
                name=display_name,
                sector="Crypto",
                asset_class="crypto",
            )
        )

    # ── Forex ────────────────────────────────────────────────────
    for symbol, display_name in FOREX_UNIVERSE.items():
        entries.append(
            SearchEntry(
                symbol=symbol,
                name=display_name,
                sector="Forex",
                asset_class="forex",
            )
        )

    # ── Commodities ──────────────────────────────────────────────
    for symbol, display_name in COMMODITY_UNIVERSE.items():
        entries.append(
            SearchEntry(
                symbol=symbol,
                name=display_name,
                sector="Commodity",
                asset_class="commodity",
            )
        )

    return tuple(entries)


# Pre-compute set of S&P 500 symbols for priority boosting
@functools.lru_cache(maxsize=1)
def _sp500_symbols() -> frozenset[str]:
    """Return frozenset of S&P 500 symbols for priority checks."""
    from quant_tools.data.universe_loader import load_universe

    return frozenset(load_universe("sp500").keys())


def search_symbols(query: str, limit: int = 10) -> list[SearchEntry]:
    """Search the index with exact > prefix > name-substring ranking.

    S&P 500 members are prioritized within each tier.

    Args:
        query: Search string (case-insensitive).
        limit: Maximum results to return.

    Returns:
        Ranked list of matching :class:`SearchEntry`.
    """
    if not query or not query.strip():
        return []

    q = query.strip().lower()
    index = build_search_index()
    sp500 = _sp500_symbols()

    exact: list[SearchEntry] = []
    prefix: list[SearchEntry] = []
    substring: list[SearchEntry] = []

    for entry in index:
        sym_lower = entry.symbol.lower()
        name_lower = entry.name.lower()

        if sym_lower == q:
            exact.append(entry)
        elif sym_lower.startswith(q):
            prefix.append(entry)
        elif q in name_lower:
            substring.append(entry)

    # Within each tier, S&P 500 members come first
    def _sp500_key(e: SearchEntry) -> int:
        return 0 if e.symbol in sp500 else 1

    exact.sort(key=_sp500_key)
    prefix.sort(key=_sp500_key)
    substring.sort(key=_sp500_key)

    results = exact + prefix + substring
    return results[:limit]
