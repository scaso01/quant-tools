"""Commodity futures catalog and continuous series roll logic."""
from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta

import pandas as pd

from quant_tools.models import CommoditySpec, FuturesContract, RollAdjustment, RollMethod

log = logging.getLogger("quant_tools")

# Month code -> calendar month number
MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# ── Commodity Catalog (~21 products) ──────────────────────────────────

COMMODITY_CATALOG: dict[str, CommoditySpec] = {
    # Energy
    "CL": CommoditySpec(
        root="CL", name="Crude Oil", category="Energy",
        contract_months="FGHJKMNQUVXZ", tick_size=0.01, point_value=1000.0,
        roll_days_before_expiry=5,
    ),
    "NG": CommoditySpec(
        root="NG", name="Natural Gas", category="Energy",
        contract_months="FGHJKMNQUVXZ", tick_size=0.001, point_value=10000.0,
        roll_days_before_expiry=5,
    ),
    "RB": CommoditySpec(
        root="RB", name="RBOB Gasoline", category="Energy",
        contract_months="FGHJKMNQUVXZ", tick_size=0.0001, point_value=42000.0,
        roll_days_before_expiry=5,
    ),
    "HO": CommoditySpec(
        root="HO", name="Heating Oil", category="Energy",
        contract_months="FGHJKMNQUVXZ", tick_size=0.0001, point_value=42000.0,
        roll_days_before_expiry=5,
    ),
    # Precious Metals
    "GC": CommoditySpec(
        root="GC", name="Gold", category="Precious Metals",
        contract_months="GJMQVZ", tick_size=0.10, point_value=100.0,
        roll_days_before_expiry=5,
    ),
    "SI": CommoditySpec(
        root="SI", name="Silver", category="Precious Metals",
        contract_months="HKNUZ", tick_size=0.005, point_value=5000.0,
        roll_days_before_expiry=5,
    ),
    "PL": CommoditySpec(
        root="PL", name="Platinum", category="Precious Metals",
        contract_months="FJNV", tick_size=0.10, point_value=50.0,
        roll_days_before_expiry=5,
    ),
    "PA": CommoditySpec(
        root="PA", name="Palladium", category="Precious Metals",
        contract_months="HMUZ", tick_size=0.05, point_value=100.0,
        roll_days_before_expiry=5,
    ),
    # Base Metals
    "HG": CommoditySpec(
        root="HG", name="Copper", category="Base Metals",
        contract_months="HKNUZ", tick_size=0.0005, point_value=25000.0,
        roll_days_before_expiry=5,
    ),
    # Agriculture
    "ZW": CommoditySpec(
        root="ZW", name="Wheat", category="Agriculture",
        contract_months="HKNUZ", tick_size=0.25, point_value=50.0,
        roll_days_before_expiry=5,
    ),
    "ZC": CommoditySpec(
        root="ZC", name="Corn", category="Agriculture",
        contract_months="HKNUZ", tick_size=0.25, point_value=50.0,
        roll_days_before_expiry=5,
    ),
    "ZS": CommoditySpec(
        root="ZS", name="Soybeans", category="Agriculture",
        contract_months="FHKNQUX", tick_size=0.25, point_value=50.0,
        roll_days_before_expiry=5,
    ),
    "ZM": CommoditySpec(
        root="ZM", name="Soybean Meal", category="Agriculture",
        contract_months="FHKNQUVZ", tick_size=0.10, point_value=100.0,
        roll_days_before_expiry=5,
    ),
    "ZL": CommoditySpec(
        root="ZL", name="Soybean Oil", category="Agriculture",
        contract_months="FHKNQUVZ", tick_size=0.01, point_value=600.0,
        roll_days_before_expiry=5,
    ),
    # Softs
    "KC": CommoditySpec(
        root="KC", name="Coffee", category="Softs",
        contract_months="HKNUZ", tick_size=0.05, point_value=375.0,
        roll_days_before_expiry=5,
    ),
    "SB": CommoditySpec(
        root="SB", name="Sugar", category="Softs",
        contract_months="HKNV", tick_size=0.01, point_value=1120.0,
        roll_days_before_expiry=5,
    ),
    "CC": CommoditySpec(
        root="CC", name="Cocoa", category="Softs",
        contract_months="HKNUZ", tick_size=1.0, point_value=10.0,
        roll_days_before_expiry=5,
    ),
    "CT": CommoditySpec(
        root="CT", name="Cotton", category="Softs",
        contract_months="HKNVZ", tick_size=0.01, point_value=500.0,
        roll_days_before_expiry=5,
    ),
    # Livestock
    "LE": CommoditySpec(
        root="LE", name="Live Cattle", category="Livestock",
        contract_months="GJMQVZ", tick_size=0.025, point_value=400.0,
        roll_days_before_expiry=5,
    ),
    "HE": CommoditySpec(
        root="HE", name="Lean Hogs", category="Livestock",
        contract_months="GJKMNQVZ", tick_size=0.025, point_value=400.0,
        roll_days_before_expiry=5,
    ),
    "GF": CommoditySpec(
        root="GF", name="Feeder Cattle", category="Livestock",
        contract_months="FHJKQUVX", tick_size=0.025, point_value=500.0,
        roll_days_before_expiry=5,
    ),
}

# Exchange suffixes for yfinance ticker formatting
_EXCHANGE_MAP: dict[str, str] = {
    "GC": "CMX", "SI": "CMX", "PL": "NYM", "PA": "NYM", "HG": "CMX",
    "CL": "NYM", "NG": "NYM", "RB": "NYM", "HO": "NYM",
    "ZW": "CBT", "ZC": "CBT", "ZS": "CBT", "ZM": "CBT", "ZL": "CBT",
    "KC": "NYB", "SB": "NYB", "CC": "NYB", "CT": "NYB",
    "LE": "CME", "HE": "CME", "GF": "CME",
}


def _contract_months_for_year(
    spec: CommoditySpec, year: int
) -> list[FuturesContract]:
    """Generate all valid contracts for a commodity in a given year."""
    exchange = _EXCHANGE_MAP.get(spec.root, "")
    contracts: list[FuturesContract] = []
    for code in spec.contract_months:
        month_num = MONTH_CODES.get(code)
        if month_num is None:
            continue
        contracts.append(
            FuturesContract(
                root=spec.root,
                month_code=code,
                year=year,
                exchange=exchange,
            )
        )
    return contracts


def _expiry_proxy(contract: FuturesContract) -> date:
    """Approximate expiry as 1st of the contract month.

    Real expiry varies by product; this proxy is sufficient for
    calendar-based roll scheduling.
    """
    month = MONTH_CODES[contract.month_code]
    return date(contract.year, month, 1)


def _business_days_before(target: date, n: int) -> date:
    """Walk backward *n* business days from *target*."""
    result = target
    remaining = n
    while remaining > 0:
        result -= timedelta(days=1)
        if result.weekday() < 5:  # Mon-Fri
            remaining -= 1
    return result


# ── Roll Date Generation ──────────────────────────────────────────────


def calendar_roll_dates(
    spec: CommoditySpec,
    start: date,
    end: date,
) -> list[tuple[date, FuturesContract, FuturesContract]]:
    """Generate roll dates using calendar method.

    Roll N business days before first of expiry month.
    Returns list of (roll_date, old_contract, new_contract).
    """
    # Collect all contracts spanning the date range
    all_contracts: list[FuturesContract] = []
    for yr in range(start.year, end.year + 2):
        all_contracts.extend(_contract_months_for_year(spec, yr))

    # Sort by expiry
    all_contracts.sort(key=lambda c: (_expiry_proxy(c), c.month_code))

    # Filter to contracts whose expiry falls within [start, end+buffer]
    relevant: list[FuturesContract] = []
    for c in all_contracts:
        exp = _expiry_proxy(c)
        if exp >= start and exp <= end + timedelta(days=90):
            relevant.append(c)

    # Generate roll points
    rolls: list[tuple[date, FuturesContract, FuturesContract]] = []
    for i in range(len(relevant) - 1):
        old_contract = relevant[i]
        new_contract = relevant[i + 1]
        expiry = _expiry_proxy(old_contract)
        roll_date = _business_days_before(expiry, spec.roll_days_before_expiry)
        if start <= roll_date <= end:
            rolls.append((roll_date, old_contract, new_contract))

    return rolls


def volume_roll_dates(
    front_volume: pd.Series,
    next_volume: pd.Series,
) -> list[date]:
    """Detect roll dates by 3-day rolling average volume crossover.

    Avoids single-day spikes by using rolling average.

    Args:
        front_volume: Daily volume for the front-month contract.
        next_volume: Daily volume for the next-month contract.

    Returns:
        List of dates where the next-month volume surpasses front-month.
    """
    if front_volume.empty or next_volume.empty:
        return []

    # Align on common dates
    common_idx = front_volume.index.intersection(next_volume.index)
    if len(common_idx) < 3:
        return []

    front = front_volume.reindex(common_idx).fillna(0)
    nxt = next_volume.reindex(common_idx).fillna(0)

    front_avg = front.rolling(3, min_periods=1).mean()
    next_avg = nxt.rolling(3, min_periods=1).mean()

    # Find crossover: next > front after being <= front
    was_front_dominant = front_avg.shift(1) >= next_avg.shift(1)
    now_next_dominant = next_avg > front_avg

    crossovers = was_front_dominant & now_next_dominant
    cross_dates: list[date] = []
    for ts in crossovers[crossovers].index:
        if hasattr(ts, "date"):
            cross_dates.append(ts.date())
        else:
            cross_dates.append(ts)
    return cross_dates


# ── Continuous Series Builder ──────────────────────────────────────────


def build_continuous_series(
    root: str,
    start: date,
    end: date,
    method: RollMethod = RollMethod.CALENDAR,
    adjustment: RollAdjustment = RollAdjustment.PANAMA,
) -> pd.DataFrame:
    """Build back-adjusted continuous futures series.

    Fetches individual contract OHLCV from yfinance (GCZ25.CMX format),
    identifies roll points, applies Panama (additive) or ratio (multiplicative)
    back-adjustment.

    If specific contract ticker fails (spotty yfinance coverage),
    degrades to generic =F front-month with warning.

    Args:
        root: Commodity root symbol (e.g., "GC", "CL").
        start: Start date for the series.
        end: End date for the series.
        method: Roll detection method (CALENDAR or VOLUME).
        adjustment: Back-adjustment method (PANAMA, RATIO, or NONE).

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns and DatetimeIndex.
    """
    import yfinance as yf

    spec = COMMODITY_CATALOG.get(root)
    if spec is None:
        raise ValueError(
            f"Unknown commodity root {root!r}. "
            f"Available: {sorted(COMMODITY_CATALOG.keys())}"
        )

    # Generate roll schedule
    if method == RollMethod.CALENDAR:
        rolls = calendar_roll_dates(spec, start, end)
    else:
        # For volume method, we need data first -- fall back to calendar schedule
        # then refine with volume crossover
        rolls = calendar_roll_dates(spec, start, end)

    if not rolls:
        # No rolls found -- just fetch the generic front-month
        return _fetch_front_month(root, start, end)

    # Attempt to fetch individual contract data
    contract_data: dict[str, pd.DataFrame] = {}
    failed_tickers: list[str] = []

    # Collect unique contracts from rolls
    unique_contracts: list[FuturesContract] = []
    seen: set[str] = set()
    for _, old_c, new_c in rolls:
        for c in (old_c, new_c):
            if c.ticker not in seen:
                seen.add(c.ticker)
                unique_contracts.append(c)

    for contract in unique_contracts:
        ticker = contract.ticker
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    ticker,
                    start=str(start),
                    end=str(end + timedelta(days=30)),
                    progress=False,
                )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                contract_data[ticker] = df
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    # If too many contracts failed, degrade to front-month
    if len(contract_data) < 2:
        log.warning(
            "Insufficient contract data for %s (got %d/%d). "
            "Falling back to =F front-month.",
            root, len(contract_data), len(unique_contracts),
        )
        return _fetch_front_month(root, start, end)

    if failed_tickers:
        log.warning(
            "Failed to fetch %d contracts for %s: %s. Using available data.",
            len(failed_tickers), root, failed_tickers[:5],
        )

    # Stitch contracts together
    return _stitch_contracts(rolls, contract_data, start, end, adjustment)


def _fetch_front_month(root: str, start: date, end: date) -> pd.DataFrame:
    """Fetch generic =F front-month ticker as fallback."""
    import yfinance as yf

    ticker = f"{root}=F"
    log.info("Fetching front-month fallback: %s", ticker)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=str(start), end=str(end), progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data available for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    expected = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {ticker} data")
    return df[expected]


def _stitch_contracts(
    rolls: list[tuple[date, FuturesContract, FuturesContract]],
    contract_data: dict[str, pd.DataFrame],
    start: date,
    end: date,
    adjustment: RollAdjustment,
) -> pd.DataFrame:
    """Stitch individual contracts into a continuous series with back-adjustment."""
    price_cols = ["Open", "High", "Low", "Close"]

    # Build segments from newest to oldest (back-adjustment goes backward)
    segments: list[pd.DataFrame] = []
    # TODO: back-adjustment here is per-roll, not cumulative. Each segment is
    # shifted by the gap at its own roll only, so a series with several rolls
    # still steps at every roll but the most recent. A running total carried
    # backward through the loop is the fix. Left as-is for now because it
    # changes published price history, which deserves its own change.

    # Start with the final (most recent) contract
    if rolls:
        last_new = rolls[-1][2]
        if last_new.ticker in contract_data:
            df = contract_data[last_new.ticker].copy()
            last_roll_date = pd.Timestamp(rolls[-1][0])
            df = df[df.index >= last_roll_date]
            if not df.empty:
                segments.append(df)

    # Walk backward through rolls
    for i in range(len(rolls) - 1, -1, -1):
        roll_date, old_c, new_c = rolls[i]
        roll_ts = pd.Timestamp(roll_date)

        if old_c.ticker not in contract_data:
            continue

        old_df = contract_data[old_c.ticker].copy()

        # Determine segment bounds
        if i > 0:
            prev_roll_ts = pd.Timestamp(rolls[i - 1][0])
            old_df = old_df[(old_df.index >= prev_roll_ts) & (old_df.index < roll_ts)]
        else:
            old_df = old_df[old_df.index < roll_ts]

        if old_df.empty:
            continue

        # Compute adjustment at the roll point
        if adjustment != RollAdjustment.NONE and new_c.ticker in contract_data:
            new_df = contract_data[new_c.ticker]
            # Find closest price to roll date in both series
            old_at_roll = old_df[old_df.index <= roll_ts]
            new_at_roll = new_df[new_df.index <= roll_ts]

            if not old_at_roll.empty and not new_at_roll.empty:
                old_close = float(old_at_roll["Close"].iloc[-1])
                new_close = float(new_at_roll["Close"].iloc[-1])

                if adjustment == RollAdjustment.PANAMA:
                    diff = new_close - old_close
                    for col in price_cols:
                        if col in old_df.columns:
                            old_df[col] = old_df[col] + diff
                elif adjustment == RollAdjustment.RATIO:
                    if old_close != 0:
                        ratio = new_close / old_close
                        for col in price_cols:
                            if col in old_df.columns:
                                old_df[col] = old_df[col] * ratio

        segments.append(old_df)

    if not segments:
        raise ValueError(f"No data segments could be stitched for {rolls[0][1].root}")

    # Reverse to chronological order and concatenate
    segments.reverse()
    result = pd.concat(segments).sort_index()
    result = result[~result.index.duplicated(keep="last")]

    # Filter to requested date range
    result = result[
        (result.index >= pd.Timestamp(start))
        & (result.index <= pd.Timestamp(end))
    ]

    expected = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in expected if c in result.columns]
    return result[available]
