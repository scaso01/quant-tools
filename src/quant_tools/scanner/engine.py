"""Concurrent stock scanner engine using ThreadPoolExecutor."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# yf.download is not thread-safe — serialize price fetches
_download_lock = threading.Lock()

from quant_tools.config import cfg
from quant_tools.data.market_data import fetch_prices, fetch_ticker_info
from quant_tools.errors import QuantToolsError
from quant_tools.data.universe import SP500_TOP100
from quant_tools.models import AssetClass, ScanResult, SignalType
from quant_tools.scanner.signals import (
    commodity_signal,
    crypto_momentum_signal,
    forex_signal,
    mean_reversion_signal,
    momentum_signal,
    quality_signal,
    value_signal,
)

log = logging.getLogger("quant_tools")

# Mapping from SignalType enum to scoring function
_SIGNAL_FUNCS: dict[SignalType, Callable] = {
    SignalType.MOMENTUM: momentum_signal,
    SignalType.VALUE: value_signal,
    SignalType.QUALITY: quality_signal,
    SignalType.MEAN_REVERSION: mean_reversion_signal,
}

# Signal functions that need price data vs info data
_PRICE_SIGNALS = {SignalType.MOMENTUM, SignalType.MEAN_REVERSION}
_INFO_SIGNALS = {SignalType.VALUE, SignalType.QUALITY}


def _prefetch_prices(
    symbols: list[str], period: str
) -> tuple[dict[str, "pd.DataFrame"], set[str]]:
    """Download daily bars for many symbols in as few requests as possible.

    Every worker used to call yf.download for one symbol while holding the
    global lock below, so the thread pool was concurrent in name only and a
    922-symbol universe took over ten minutes of round trips. yfinance accepts a
    list and fetches them together, which turns that into a handful of requests.

    Returns ``(frames, covered)``. ``covered`` is every symbol that was part of a
    request that came back, whether or not it had data, so the caller can tell
    "Yahoo has nothing for this ticker" apart from "this request never ran" and
    skip re-asking for symbols already known to be dead.
    """
    import pandas as pd
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    covered: set[str] = set()
    # Chunked so one enormous URL cannot fail the whole scan, and so a partial
    # failure costs one chunk rather than everything.
    chunk_size = 100
    requests = 0
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        requests += 1
        try:
            with _download_lock:
                raw = yf.download(
                    chunk, period=period, group_by="ticker",
                    progress=False, threads=True, auto_adjust=True,
                )
        except Exception:
            log.warning("Batch price prefetch failed for %d symbols", len(chunk), exc_info=True)
            continue

        if raw is None or raw.empty:
            continue

        covered.update(chunk)
        for symbol in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol not in raw.columns.get_level_values(0):
                        continue
                    df = raw[symbol]
                else:
                    # A single-symbol chunk comes back without the ticker level.
                    df = raw
                df = df.dropna(how="all")
                if not df.empty and "Close" in df.columns:
                    out[symbol] = df
            except Exception:
                log.debug("Could not split prefetched frame for %s", symbol, exc_info=True)

    dead = len(covered) - len(out)
    log.info(
        "Prefetched %d/%d symbols in %d request(s); %d had no data and will be skipped",
        len(out), len(symbols), requests, dead,
    )
    return out, covered


def _scan_single(
    symbol: str,
    signals: list[SignalType],
    period: str = "1y",
    asset_class: AssetClass | None = None,
    interval: str = "1d",
    prefetched: "pd.DataFrame | None" = None,
    known_empty: bool = False,
) -> ScanResult | None:
    """Scan a single symbol. Returns None on failure.

    Args:
        symbol: Ticker symbol to scan.
        signals: List of signal types to compute.
        period: Price history period (default "1y").
        asset_class: Optional asset class. When non-EQUITY, uses
            asset-specific signal functions.
        prefetched: Bars already downloaded in bulk. When given, the per-symbol
            download is skipped.
        known_empty: The bulk download covered this symbol and Yahoo returned
            nothing for it, so there is no point asking again.

    Returns:
        ScanResult or None if the symbol could not be scanned.
    """
    if known_empty:
        # Delisted tickers linger in the bigger index lists. Retrying each one
        # individually cost three attempts with exponential backoff apiece,
        # which was most of the wall clock on a Russell 1000 scan.
        log.debug("Skipping %s: bulk download returned no data", symbol)
        return None

    try:
        needs_prices = any(s in _PRICE_SIGNALS for s in signals)
        needs_info = any(s in _INFO_SIGNALS for s in signals)

        close: np.ndarray | None = None
        volume: np.ndarray | None = None
        info: dict = {}
        price = 0.0
        change_1d = 0.0
        change_5d = 0.0
        change_1m = 0.0
        rsi_14: float | None = None

        if needs_prices:
            if asset_class and asset_class != AssetClass.EQUITY:
                from quant_tools.data.providers.registry import ProviderRegistry
                registry = ProviderRegistry()
                with _download_lock:
                    df = registry.fetch_ohlcv(symbol, period=period, interval=interval)
            elif interval != "1d":
                from quant_tools.data.providers.registry import ProviderRegistry
                registry = ProviderRegistry()
                with _download_lock:
                    df = registry.fetch_ohlcv(symbol, period=period, interval=interval)
            elif prefetched is not None:
                df = prefetched
            else:
                with _download_lock:
                    df = fetch_prices(symbol, period=period)
            close = df["Close"].values.astype(np.float64)
            if "Volume" in df.columns:
                volume = df["Volume"].values.astype(np.float64)

            if len(close) >= 1:
                price = float(close[-1])
            if len(close) >= 2:
                change_1d = (close[-1] / close[-2] - 1) * 100
            if len(close) >= 5:
                change_5d = (close[-1] / close[-5] - 1) * 100
            if len(close) >= 21:
                change_1m = (close[-1] / close[-21] - 1) * 100

            from quant_tools.data.indicators import rsi as calc_rsi
            rsi_vals = calc_rsi(close, period=14)
            valid_rsi = rsi_vals[~np.isnan(rsi_vals)]
            if len(valid_rsi) > 0:
                rsi_14 = round(float(valid_rsi[-1]), 2)

        if needs_info and (asset_class is None or asset_class == AssetClass.EQUITY):
            info = fetch_ticker_info(symbol)

        # Compute signal scores
        is_equity = asset_class is None or asset_class == AssetClass.EQUITY
        signal_scores: dict[str, float] = {}
        breakdowns: dict[str, dict[str, float]] = {}
        for sig in signals:
            if sig == SignalType.MOMENTUM and close is not None:
                if is_equity:
                    score, bd = momentum_signal(close, volume, return_breakdown=True)
                    signal_scores[sig.value] = score
                    breakdowns[sig.value] = bd
                elif asset_class == AssetClass.CRYPTO:
                    signal_scores[sig.value] = crypto_momentum_signal(close, volume)
                elif asset_class == AssetClass.FOREX:
                    signal_scores[sig.value] = forex_signal(close)
                elif asset_class == AssetClass.COMMODITY:
                    signal_scores[sig.value] = commodity_signal(close, volume)
            elif sig == SignalType.MEAN_REVERSION and close is not None:
                if is_equity:
                    score, bd = mean_reversion_signal(close, return_breakdown=True)
                    signal_scores[sig.value] = score
                    breakdowns[sig.value] = bd
                elif asset_class == AssetClass.COMMODITY:
                    signal_scores[sig.value] = commodity_signal(close, volume)
                else:
                    signal_scores[sig.value] = 50.0
            elif sig == SignalType.VALUE:
                if is_equity:
                    score, bd = value_signal(info, return_breakdown=True)
                    signal_scores[sig.value] = score
                    breakdowns[sig.value] = bd
                else:
                    signal_scores[sig.value] = 50.0
            elif sig == SignalType.QUALITY:
                if is_equity:
                    score, bd = quality_signal(info, return_breakdown=True)
                    signal_scores[sig.value] = score
                    breakdowns[sig.value] = bd
                else:
                    signal_scores[sig.value] = 50.0

        if not signal_scores:
            return None

        composite = float(np.mean(list(signal_scores.values())))

        # Determine primary signal (highest scoring)
        primary_signal_name = max(signal_scores, key=signal_scores.get)  # type: ignore[arg-type]
        primary_signal = SignalType(primary_signal_name)

        # Volume ratio
        vol_ratio: float | None = None
        if volume is not None and len(volume) >= 20:
            avg_vol = float(np.mean(volume[-20:]))
            if avg_vol > 0:
                vol_ratio = round(float(volume[-1]) / avg_vol, 2)

        name = info.get("name", symbol) if info else symbol
        if is_equity:
            sector = SP500_TOP100.get(symbol, info.get("sector", "Unknown"))
        else:
            sector = asset_class.value.capitalize() if asset_class else "Unknown"

        details: dict = {**signal_scores}
        if breakdowns:
            details["breakdown"] = breakdowns

        return ScanResult(
            symbol=symbol,
            name=str(name),
            sector=str(sector),
            signal=primary_signal,
            score=round(composite, 2),
            price=round(price, 2),
            change_1d=round(change_1d, 2),
            change_5d=round(change_5d, 2),
            change_1m=round(change_1m, 2),
            rsi_14=rsi_14,
            volume_ratio=vol_ratio,
            details=details,
        )

    except QuantToolsError as exc:
        # A delisted ticker, or a pair the exchange does not list, is an
        # ordinary outcome when scanning a broad universe. Each one used to
        # print a full stack trace, so `scan --asset-class crypto` buried its
        # own results under fifteen of them.
        log.debug("Skipping %s: %s", symbol, exc)
        return None
    except Exception:
        log.warning("Failed to scan %s", symbol, exc_info=True)
        return None


def scan_universe(
    symbols: list[str],
    signals: list[SignalType] | None = None,
    max_workers: int | None = None,
    top_n: int = 20,
    period: str = "1y",
    asset_class: AssetClass | None = None,
    interval: str = "1d",
) -> list[ScanResult]:
    """Scan symbols concurrently, rank by composite score.

    Args:
        symbols: List of ticker symbols to scan.
        signals: Signal types to compute. Defaults to all signals.
        max_workers: Max threads. Defaults to config value.
        top_n: Number of top results to return.
        period: Price history period (default "1y").
        asset_class: Optional asset class. When non-EQUITY, uses
            asset-specific signal functions.

    Returns:
        List of ScanResult sorted by score descending, limited to top_n.
    """
    if not symbols:
        return []

    if signals is None:
        signals = list(SignalType)

    if max_workers is None:
        max_workers = cfg.scanner_max_workers

    results: list[ScanResult] = []

    # Only the plain equity daily path goes through fetch_prices; the other
    # branches use the provider registry, which this cannot batch.
    prefetched: dict = {}
    covered: set[str] = set()
    needs_prices = any(s in _PRICE_SIGNALS for s in signals)
    if needs_prices and interval == "1d" and asset_class in (None, AssetClass.EQUITY):
        prefetched, covered = _prefetch_prices(symbols, period)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                _scan_single, sym, signals, period, asset_class, interval,
                prefetched.get(sym), sym in covered and sym not in prefetched,
            ): sym
            for sym in symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                log.warning("Scanner future failed for %s", symbol, exc_info=True)

    # Sort by composite score descending
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_n]


def scan_multi_asset(
    assets: list[tuple[str, AssetClass]],
    signals: list[SignalType] | None = None,
    max_workers: int | None = None,
    top_n: int = 20,
    period: str = "1y",
    interval: str = "1d",
) -> list[ScanResult]:
    """Scan a mixed list of (symbol, asset_class) tuples.

    Groups symbols by asset class, scans each group, and merges results.

    Args:
        assets: List of (symbol, asset_class) tuples.
        signals: Signal types to compute. Defaults to all signals.
        max_workers: Max threads. Defaults to config value.
        top_n: Number of top results to return.
        period: Price history period (default "1y").

    Returns:
        List of ScanResult sorted by score descending, limited to top_n.
    """
    # Group by asset class
    groups: dict[AssetClass, list[str]] = {}
    for symbol, ac in assets:
        groups.setdefault(ac, []).append(symbol)

    all_results: list[ScanResult] = []
    for ac, symbols in groups.items():
        results = scan_universe(
            symbols=symbols,
            signals=signals,
            max_workers=max_workers,
            top_n=len(symbols),  # Don't truncate per-group
            period=period,
            asset_class=ac,
            interval=interval,
        )
        all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_n]
