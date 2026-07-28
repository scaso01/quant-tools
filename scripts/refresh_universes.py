#!/usr/bin/env python3
"""Refresh universe JSON files from Wikipedia.

Scrapes S&P 500 and Russell 1000 constituent lists from Wikipedia,
normalizes GICS sector/sub-industry names, and writes updated JSON
to src/quant_tools/data/universes/.

Usage:
    python scripts/refresh_universes.py [--sp500] [--russell1000] [--dry-run]

    With no flags, refreshes both S&P 500 and Russell 1000.
    International universe is maintained manually (ADR lists change rarely).

Requires: pandas, lxml (for pandas.read_html)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas lxml")
    sys.exit(1)

UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "data" / "universes"

# Wikipedia URLs
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
RUSSELL1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"


def scrape_sp500() -> dict[str, dict[str, str]]:
    """Scrape S&P 500 constituents from Wikipedia.

    Returns:
        Dict mapping ticker -> {name, sector, sub_industry, region}.
    """
    print("Fetching S&P 500 from Wikipedia...")
    tables = pd.read_html(SP500_URL, header=0)
    # First table is the constituent list
    df = tables[0]

    # Column names vary; try common patterns
    ticker_col = _find_column(df, ["Symbol", "Ticker symbol", "Ticker"])
    name_col = _find_column(df, ["Security", "Company", "Name"])
    sector_col = _find_column(df, ["GICS Sector", "Sector"])
    sub_col = _find_column(df, ["GICS Sub-Industry", "Sub-Industry", "Sub Industry"])

    universe: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-")
        name = str(row[name_col]).strip()
        sector = str(row[sector_col]).strip()
        sub_industry = str(row[sub_col]).strip() if sub_col else ""

        if ticker and ticker != "nan" and name != "nan":
            universe[ticker] = {
                "name": name,
                "sector": sector,
                "sub_industry": sub_industry,
                "region": "US",
            }

    print(f"  Found {len(universe)} S&P 500 constituents")
    return universe


def scrape_russell1000() -> dict[str, dict[str, str]]:
    """Scrape Russell 1000 constituents from Wikipedia.

    Note: Wikipedia's Russell 1000 page may not have a full constituent
    table. If scraping fails, falls back to the existing JSON + S&P 500
    overlay.

    Returns:
        Dict mapping ticker -> {name, sector, sub_industry, region}.
    """
    print("Fetching Russell 1000 from Wikipedia...")
    try:
        tables = pd.read_html(RUSSELL1000_URL, header=0)
    except Exception as exc:
        print(f"  WARNING: Could not scrape Russell 1000: {exc}")
        print("  Falling back to existing russell1000.json")
        return _load_existing("russell1000")

    # Find the largest table (most likely the constituent list)
    if not tables:
        print("  WARNING: No tables found on Russell 1000 page")
        return _load_existing("russell1000")

    df = max(tables, key=len)

    ticker_col = _find_column(df, ["Ticker", "Symbol", "Ticker symbol"])
    name_col = _find_column(df, ["Company", "Security", "Name"])

    if ticker_col is None or name_col is None:
        print("  WARNING: Could not identify ticker/name columns")
        print(f"  Available columns: {list(df.columns)}")
        return _load_existing("russell1000")

    # Russell 1000 Wikipedia table may lack sector data
    sector_col = _find_column(df, ["GICS Sector", "Sector"])
    sub_col = _find_column(df, ["GICS Sub-Industry", "Sub-Industry", "Sub Industry"])

    universe: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-")
        name = str(row[name_col]).strip()
        sector = str(row[sector_col]).strip() if sector_col else ""
        sub_industry = str(row[sub_col]).strip() if sub_col else ""

        if ticker and ticker != "nan" and name != "nan":
            universe[ticker] = {
                "name": name,
                "sector": sector,
                "sub_industry": sub_industry,
                "region": "US",
            }

    # If we got very few results, merge with existing
    if len(universe) < 500:
        print(f"  Only found {len(universe)} entries, merging with existing JSON")
        existing = _load_existing("russell1000")
        existing.update(universe)  # scraped data overrides existing
        universe = existing

    print(f"  Found {len(universe)} Russell 1000 constituents")
    return universe


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from candidates."""
    for name in candidates:
        if name in df.columns:
            return name
        # Case-insensitive fallback
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None


def _load_existing(name: str) -> dict[str, dict[str, str]]:
    """Load existing universe JSON as raw dicts."""
    path = UNIVERSES_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_universe(name: str, data: dict[str, dict[str, str]], *, dry_run: bool = False) -> None:
    """Write universe dict to JSON file."""
    path = UNIVERSES_DIR / f"{name}.json"
    if dry_run:
        print(f"  [DRY RUN] Would write {len(data)} entries to {path}")
        return
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {len(data)} entries to {path}")


def _diff_report(name: str, new_data: dict[str, dict[str, str]]) -> None:
    """Print a diff summary between existing and new data."""
    existing = _load_existing(name)
    added = set(new_data.keys()) - set(existing.keys())
    removed = set(existing.keys()) - set(new_data.keys())
    if added:
        print(f"  + {len(added)} added: {', '.join(sorted(added)[:10])}{'...' if len(added) > 10 else ''}")
    if removed:
        print(f"  - {len(removed)} removed: {', '.join(sorted(removed)[:10])}{'...' if len(removed) > 10 else ''}")
    if not added and not removed:
        print("  No changes to constituents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh universe JSON from Wikipedia")
    parser.add_argument("--sp500", action="store_true", help="Refresh S&P 500 only")
    parser.add_argument("--russell1000", action="store_true", help="Refresh Russell 1000 only")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    # If no specific flag, refresh both
    do_sp500 = args.sp500 or (not args.sp500 and not args.russell1000)
    do_russell = args.russell1000 or (not args.sp500 and not args.russell1000)

    if do_sp500:
        sp500 = scrape_sp500()
        _diff_report("sp500", sp500)
        _write_universe("sp500", sp500, dry_run=args.dry_run)

    if do_russell:
        russell = scrape_russell1000()
        _diff_report("russell1000", russell)
        _write_universe("russell1000", russell, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
