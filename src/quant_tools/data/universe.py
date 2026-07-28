"""Universe definitions, sector mappings, and preset populations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_tools.models import TickerMeta

# ── Top 100 S&P 500 Constituents by Market Cap ─────────────────────────
# Mapping: ticker -> GICS sector name
# All 11 GICS sectors represented

SP500_TOP100: dict[str, str] = {
    # Technology (14)
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "CSCO": "Technology",
    "ACN": "Technology",
    "ADBE": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "IBM": "Technology",
    "INTU": "Technology",
    "TXN": "Technology",
    # Communication Services (6)
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "TMUS": "Communication Services",
    # Consumer Discretionary (10)
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary",
    # Consumer Staples (8)
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PM": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "CL": "Consumer Staples",
    # Financials (12)
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "BLK": "Financials",
    "SPGI": "Financials",
    "AXP": "Financials",
    "C": "Financials",
    # Healthcare (12)
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "PFE": "Healthcare",
    "DHR": "Healthcare",
    "AMGN": "Healthcare",
    "BMY": "Healthcare",
    "ISRG": "Healthcare",
    # Industrials (10)
    "GE": "Industrials",
    "CAT": "Industrials",
    "RTX": "Industrials",
    "UNP": "Industrials",
    "HON": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "UPS": "Industrials",
    "MMM": "Industrials",
    # Energy (8)
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "OXY": "Energy",
    # Materials (6)
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "FCX": "Materials",
    "NEM": "Materials",
    "ECL": "Materials",
    # Utilities (6)
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "SRE": "Utilities",
    # Real Estate (8)
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "EQIX": "Real Estate",
    "CCI": "Real Estate",
    "PSA": "Real Estate",
    "O": "Real Estate",
    "SPG": "Real Estate",
    "WELL": "Real Estate",
}


# ── Derived Sector Map ──────────────────────────────────────────────────

def _build_sector_map() -> dict[str, list[str]]:
    """Build reverse mapping from sector -> list of tickers."""
    sector_map: dict[str, list[str]] = defaultdict(list)
    for ticker, sector in SP500_TOP100.items():
        sector_map[sector].append(ticker)
    return dict(sector_map)


SECTOR_MAP: dict[str, list[str]] = _build_sector_map()


# ── Universe Presets ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class UniversePreset:
    """A named collection of ticker symbols for scanning."""

    name: str
    symbols: tuple[str, ...]
    description: str
    has_sectors: bool = True

    @property
    def count(self) -> int:
        return len(self.symbols)


UNIVERSE_PRESETS: dict[str, UniversePreset] = {
    "S&P 500 Top 100": UniversePreset(
        name="S&P 500 Top 100",
        symbols=tuple(SP500_TOP100.keys()),
        description="Top 100 S&P 500 by market cap — all 11 GICS sectors",
        has_sectors=True,
    ),
    "Magnificent 7": UniversePreset(
        name="Magnificent 7",
        symbols=("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"),
        description="7 largest tech & growth mega-caps",
        has_sectors=False,
    ),
    "Mega Cap 30": UniversePreset(
        name="Mega Cap 30",
        symbols=(
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            "BRK-B", "LLY", "AVGO", "JPM", "UNH", "V", "MA", "COST",
            "WMT", "HD", "NFLX", "JNJ", "ABBV", "ORCL", "CRM", "PG",
            "BAC", "MRK", "XOM", "KO", "PEP", "TMO", "CVX",
        ),
        description="Top 30 U.S. stocks by market capitalization",
        has_sectors=True,
    ),
    "Dow 30": UniversePreset(
        name="Dow 30",
        symbols=(
            "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO",
            "CVX", "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
            "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA",
            "PG", "SHW", "TRV", "UNH", "V", "WMT",
        ),
        description="Dow Jones Industrial Average — 30 blue-chip stocks",
        has_sectors=True,
    ),
    "Growth & Tech": UniversePreset(
        name="Growth & Tech",
        symbols=(
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            "AVGO", "CRM", "ADBE", "AMD", "NFLX", "ORCL", "INTU",
            "ACN", "TXN", "CSCO", "IBM", "BKNG", "ABNB", "ISRG",
            "TMUS", "INTC", "NOW", "PANW",
        ),
        description="High-growth technology & innovation leaders",
        has_sectors=False,
    ),
    "Defensive & Income": UniversePreset(
        name="Defensive & Income",
        symbols=(
            "JNJ", "PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ",
            "PM", "ABT", "MRK", "PFE", "AMGN", "BMY", "NEE", "DUK",
            "SO", "D", "AEP", "SRE", "PLD", "AMT", "O", "PSA", "WELL",
        ),
        description="Staples, utilities, healthcare & REITs for stability",
        has_sectors=False,
    ),
}


# ── Lazy-loaded Extended Universe Presets ──────────────────────────────

def _load_extended_preset(name: str) -> UniversePreset:
    """Build a UniversePreset from a JSON universe file (lazy-loaded)."""
    from quant_tools.data.universe_loader import load_universe
    universe = load_universe(name)
    return UniversePreset(
        name=name,
        symbols=tuple(sorted(universe.keys())),
        description=f"Full {name} universe from JSON ({len(universe)} constituents)",
        has_sectors=True,
    )


def _load_region_preset(region: str, preset_name: str, description: str) -> UniversePreset:
    """Build a UniversePreset filtered by region from international.json."""
    from quant_tools.data.universe_loader import load_universe
    intl = load_universe("international")
    symbols = tuple(sorted(
        sym for sym, meta in intl.items() if meta.region == region
    ))
    return UniversePreset(
        name=preset_name,
        symbols=symbols,
        description=description,
        has_sectors=True,
    )


def _build_eafe_preset() -> UniversePreset:
    """Build EAFE preset (Europe, Australasia, Far East = non-Emerging international)."""
    from quant_tools.data.universe_loader import load_universe
    intl = load_universe("international")
    eafe_regions = {"Europe", "Asia Pacific"}
    symbols = tuple(sorted(
        sym for sym, meta in intl.items() if meta.region in eafe_regions
    ))
    return UniversePreset(
        name="International EAFE",
        symbols=symbols,
        description=f"Developed market ADRs — Europe, Australasia & Far East ({len(symbols)} constituents)",
        has_sectors=True,
    )


def _load_global_preset() -> UniversePreset:
    """Build a UniversePreset combining all equity universes."""
    from quant_tools.data.universe_loader import load_combined
    combined = load_combined(["sp500", "russell1000", "international"])
    return UniversePreset(
        name="Global All",
        symbols=tuple(sorted(combined.keys())),
        description=f"All equities: US + international ({len(combined)} constituents)",
        has_sectors=True,
    )


# Extended preset builders keyed by preset name.
# Values are callables so JSON is only loaded when a preset is first accessed.
_EXTENDED_PRESET_BUILDERS: dict[str, object] = {
    "S&P 500": lambda: _load_extended_preset("sp500"),
    "Russell 1000": lambda: _load_extended_preset("russell1000"),
    "International EAFE": lambda: _build_eafe_preset(),
    "Emerging Markets": lambda: _load_region_preset(
        region="Emerging",
        preset_name="Emerging Markets",
        description="Emerging market ADRs (China, India, Brazil, etc.)",
    ),
    "Global All": _load_global_preset,
}

# Cache for materialized extended presets
_extended_cache: dict[str, UniversePreset] = {}


def _get_extended_preset(name: str) -> UniversePreset | None:
    """Get an extended preset by name, materializing on first access."""
    if name in _extended_cache:
        return _extended_cache[name]
    builder = _EXTENDED_PRESET_BUILDERS.get(name)
    if builder is None:
        return None
    preset = builder()
    _extended_cache[name] = preset
    return preset


# ── Multi-Asset Universes ──────────────────────────────────────────────

CRYPTO_UNIVERSE: dict[str, str] = {
    "BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum", "BNB/USDT": "Binance Coin",
    "SOL/USDT": "Solana", "XRP/USDT": "Ripple", "ADA/USDT": "Cardano",
    "AVAX/USDT": "Avalanche", "DOT/USDT": "Polkadot", "DOGE/USDT": "Dogecoin",
    "MATIC/USDT": "Polygon", "LINK/USDT": "Chainlink", "UNI/USDT": "Uniswap",
    "ATOM/USDT": "Cosmos", "LTC/USDT": "Litecoin", "FIL/USDT": "Filecoin",
    "NEAR/USDT": "NEAR Protocol", "APT/USDT": "Aptos", "ARB/USDT": "Arbitrum",
    "OP/USDT": "Optimism", "INJ/USDT": "Injective",
}

FOREX_UNIVERSE: dict[str, str] = {
    "EUR/USD": "Euro/US Dollar", "GBP/USD": "British Pound/US Dollar",
    "USD/JPY": "US Dollar/Japanese Yen", "AUD/USD": "Australian Dollar/US Dollar",
    "USD/CAD": "US Dollar/Canadian Dollar", "USD/CHF": "US Dollar/Swiss Franc",
    "NZD/USD": "New Zealand Dollar/US Dollar", "EUR/GBP": "Euro/British Pound",
}

COMMODITY_UNIVERSE: dict[str, str] = {
    # Precious Metals
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "PL=F": "Platinum Futures",
    "PA=F": "Palladium Futures",
    # Energy
    "CL=F": "Crude Oil Futures",
    "NG=F": "Natural Gas Futures",
    "RB=F": "RBOB Gasoline Futures",
    "HO=F": "Heating Oil Futures",
    # Base Metals
    "HG=F": "Copper Futures",
    # Agriculture
    "ZW=F": "Wheat Futures",
    "ZC=F": "Corn Futures",
    "ZS=F": "Soybean Futures",
    "ZM=F": "Soybean Meal Futures",
    "ZL=F": "Soybean Oil Futures",
    # Softs
    "KC=F": "Coffee Futures",
    "SB=F": "Sugar Futures",
    "CC=F": "Cocoa Futures",
    "CT=F": "Cotton Futures",
    # Livestock
    "LE=F": "Live Cattle Futures",
    "HE=F": "Lean Hogs Futures",
    "GF=F": "Feeder Cattle Futures",
}

COMMODITY_PRESETS: dict[str, list[str]] = {
    "All Commodities": list(COMMODITY_UNIVERSE.keys()),
    "Energy": ["CL=F", "NG=F", "RB=F", "HO=F"],
    "Precious Metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
    "Base Metals": ["HG=F"],
    "Agriculture": ["ZW=F", "ZC=F", "ZS=F", "ZM=F", "ZL=F"],
    "Softs": ["KC=F", "SB=F", "CC=F", "CT=F"],
    "Livestock": ["LE=F", "HE=F", "GF=F"],
}

# Mapping from asset_class name to universe dict
_ASSET_CLASS_UNIVERSES: dict[str, dict[str, str]] = {
    "crypto": CRYPTO_UNIVERSE,
    "forex": FOREX_UNIVERSE,
    "commodity": COMMODITY_UNIVERSE,
}


# ── Public API ──────────────────────────────────────────────────────────


def get_universe(name: str = "sp500", asset_class: str | None = None) -> list[str]:
    """Return list of ticker symbols for a named universe.

    Args:
        name: Universe name. Supports "sp500" (legacy) and any key
              from :data:`UNIVERSE_PRESETS`.
        asset_class: Optional asset class override. When provided,
            returns the corresponding multi-asset universe
            ("crypto", "forex", "commodity").

    Returns:
        List of ticker symbols.

    Raises:
        ValueError: If the universe name is not recognized.
    """
    if asset_class is not None:
        ac_lower = asset_class.lower()
        universe = _ASSET_CLASS_UNIVERSES.get(ac_lower)
        if universe is not None:
            return list(universe.keys())
        available_ac = list(_ASSET_CLASS_UNIVERSES.keys())
        raise ValueError(f"Unknown asset_class: {asset_class!r}. Available: {available_ac}")

    if name.lower() == "sp500":
        return list(SP500_TOP100.keys())
    preset = UNIVERSE_PRESETS.get(name)
    if preset is not None:
        return list(preset.symbols)
    extended = _get_extended_preset(name)
    if extended is not None:
        return list(extended.symbols)
    available = ["sp500"] + list(UNIVERSE_PRESETS.keys()) + list(_EXTENDED_PRESET_BUILDERS.keys())
    raise ValueError(f"Unknown universe: {name!r}. Available: {available}")


def get_sector_tickers(sector: str) -> list[str]:
    """Return tickers belonging to a specific sector.

    Args:
        sector: GICS sector name (e.g., "Technology", "Financials").

    Returns:
        List of ticker symbols in that sector.

    Raises:
        ValueError: If the sector is not recognized.
    """
    tickers = SECTOR_MAP.get(sector)
    if tickers is None:
        available = sorted(SECTOR_MAP.keys())
        raise ValueError(f"Unknown sector: {sector!r}. Available: {available}")
    return list(tickers)


def get_ticker_sector(symbol: str) -> str:
    """Return the sector for a given ticker symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").

    Returns:
        Sector name string.

    Raises:
        KeyError: If the symbol is not in the universe.
    """
    sector = SP500_TOP100.get(symbol.upper())
    if sector is None:
        raise KeyError(f"Symbol {symbol!r} not in SP500_TOP100 universe")
    return sector


def filter_by_region(
    universe_name: str,
    region: str,
) -> dict[str, TickerMeta]:
    """Filter a universe by geographic region.

    Args:
        universe_name: Name of the universe JSON file (e.g., "international").
        region: Region string to match (e.g., "US", "Europe", "Asia Pacific", "Emerging").

    Returns:
        Dict of symbol -> TickerMeta for matching tickers.
    """
    from quant_tools.data.universe_loader import load_universe

    universe = load_universe(universe_name)
    return {
        sym: meta for sym, meta in universe.items()
        if meta.region == region
    }


def list_all_presets() -> list[str]:
    """Return names of all available universe presets (static + extended).

    Returns:
        Sorted list of preset name strings.
    """
    static = list(UNIVERSE_PRESETS.keys())
    extended = list(_EXTENDED_PRESET_BUILDERS.keys())
    return sorted(set(static + extended + ["sp500"]))
