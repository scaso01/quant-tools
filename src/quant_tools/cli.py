"""Unified CLI dispatcher: quant-tools <command>."""

from __future__ import annotations

import sys

from quant_tools.errors import DataError, QuantToolsError
from quant_tools.logging_config import setup_logging


def main() -> None:
    """Main entry point for the quant-tools CLI."""
    setup_logging()

    if len(sys.argv) < 2:
        _print_help()
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    commands = {
        "scan": _cmd_scan,
        "backtest": _cmd_backtest,
        "portfolio": _cmd_portfolio,
        "factors": _cmd_factors,
        "predict": _cmd_predict,
        "track": _cmd_track,
        "resolve": _cmd_resolve,
        "clear": _cmd_clear,
        "filings": _cmd_filings,
        "sentiment": _cmd_sentiment,
        "macro": _cmd_macro,
        "dashboard": _cmd_dashboard,
        "web": _cmd_web,
        # Phase 1: Alternative data
        "news": _cmd_news,
        "trends": _cmd_trends,
        "defi": _cmd_defi,
        "fear-greed": _cmd_fear_greed,
        "options": _cmd_options,
        # Phase 2: Predictions
        "benchmark-models": _cmd_benchmark_models,
        # Phase 4: Risk
        "risk": _cmd_risk,
        "stress-test": _cmd_stress_test,
        "regime": _cmd_regime,
        # Phase 6: LLM
        "analyze": _cmd_analyze,
        "ask": _cmd_ask,
        # Phase 7: Reporting
        "tearsheet": _cmd_tearsheet,
        # Phase 9: Execution
        "paper-trade": _cmd_paper_trade,
        "help": lambda _: _print_help(),
        "--help": lambda _: _print_help(),
        "-h": lambda _: _print_help(),
    }

    handler = commands.get(command)
    if handler is None:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)

    try:
        rc = handler(args)
    except QuantToolsError as exc:
        # A ticker with no filings, or a missing REDDIT_CLIENT_ID, is an ordinary
        # outcome for this tool rather than a crash. These used to reach the user
        # as a raw traceback. Print the message the exception already carries and
        # exit non-zero so a script can still tell that the command failed.
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        sys.exit(130)

    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)


def _wants_help(args: list[str]) -> bool:
    """True when the user asked for help on a command argparse does not handle."""
    return any(a in ("--help", "-h") for a in args)


def _parse_holdings(raw: str) -> dict[str, float]:
    """Parse and sanity check a --holdings JSON dict.

    Shared by risk, stress-test, tearsheet and ask so they reject the same bad
    input the same way.
    """
    import json

    try:
        holdings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON for --holdings: {exc}") from exc
    if not isinstance(holdings, dict) or not holdings:
        raise DataError("--holdings must be a non-empty JSON object of symbol:weight")
    total = sum(float(v) for v in holdings.values())
    if abs(total - 1.0) > 0.01:
        raise DataError(
            f"--holdings weights sum to {total:.2f}, expected 1.0. Give each "
            'symbol its fraction of the portfolio, for example \'{"AAPL":0.6,"BND":0.4}\'.'
        )
    return {str(k): float(v) for k, v in holdings.items()}


def _portfolio_returns(holdings: dict[str, float], period: str = "1y"):
    """Weighted daily return series for a set of holdings.

    The risk, stress-test and tearsheet commands all need this and all build it
    the same way the portfolio command does.
    """
    import numpy as np

    from quant_tools.data.market_data import fetch_prices

    returns_arrays = []
    for symbol in holdings:
        df = fetch_prices(symbol, period=period)
        close = df["Close"].values.astype(float)
        returns_arrays.append(np.diff(close) / close[:-1])

    min_len = min(len(r) for r in returns_arrays)
    matrix = np.column_stack([r[:min_len] for r in returns_arrays])
    return matrix @ np.array([holdings[s] for s in holdings])


# These four sub-CLIs already return 1 when they reject their input, but the
# wrappers dropped the value, so `quant-tools scan --universe nonsense` printed
# an error and still exited 0. Anything scripting this tool saw success.
def _cmd_scan(args: list[str]) -> int:
    """Run the market scanner."""
    from quant_tools.scanner.cli import main as scan_main
    return scan_main(args)


def _cmd_backtest(args: list[str]) -> None:
    """Run a backtest."""
    # Unlike the others this returns a BacktestResult rather than an exit code;
    # it signals failure by raising, which main() turns into a non-zero exit.
    from quant_tools.backtest.cli import backtest_command
    backtest_command(args)


def _cmd_portfolio(args: list[str]) -> int:
    """Run portfolio analysis."""
    from quant_tools.portfolio.cli import main as portfolio_main
    return portfolio_main(args)


def _cmd_factors(args: list[str]) -> int:
    """Run factor analysis."""
    from quant_tools.factors.cli import main as factors_main
    return factors_main(args)


def _cmd_predict(args: list[str]) -> int:
    """Run ML ensemble prediction."""
    from quant_tools.predict.cli import cmd_predict
    return cmd_predict(args)


def _cmd_track(args: list[str]) -> int:
    """View stored predictions and performance."""
    from quant_tools.predict.cli import cmd_track
    return cmd_track(args)


def _cmd_resolve(args: list[str]) -> int:
    """Resolve pending predictions."""
    from quant_tools.predict.cli import cmd_resolve
    return cmd_resolve(args)


def _cmd_clear(args: list[str]) -> int:
    """Clear stored predictions."""
    from quant_tools.predict.cli import cmd_clear
    return cmd_clear(args)


def _cmd_filings(args: list[str]) -> None:
    """Fetch SEC EDGAR filings."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools filings",
        description="Download SEC EDGAR filings for a ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--type", default="10-K", choices=["10-K", "10-Q", "8-K"],
        help="Filing type (default: 10-K)",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of recent filings (default: 5)",
    )
    parsed = parser.parse_args(args)

    from quant_tools.data.providers.sec_edgar import SecEdgarClient

    client = SecEdgarClient()
    filings = client.fetch_filings(
        parsed.ticker, filing_type=parsed.type, num_filings=parsed.count,
    )
    for i, f in enumerate(filings, 1):
        print(f"\n{'='*60}")
        print(f"Filing {i}: {f.filing_type} — {f.filename}")
        print(f"{'='*60}")
        # Show first 500 chars as preview
        print(f.content[:500])
        if len(f.content) > 500:
            print(f"\n... ({len(f.content):,} chars total)")


def _cmd_sentiment(args: list[str]) -> None:
    """Fetch Reddit sentiment for a ticker."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools sentiment",
        description="Fetch Reddit sentiment for a stock ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--subreddits", nargs="+",
        default=None,
        help="Subreddits to search (default: wallstreetbets stocks investing)",
    )
    parsed = parser.parse_args(args)

    from quant_tools.data.sentiment.reddit_client import RedditSentimentClient

    client = RedditSentimentClient()
    result = client.fetch_sentiment(parsed.ticker, subreddits=parsed.subreddits)

    label = "Bullish" if result.score > 0.1 else ("Bearish" if result.score < -0.1 else "Neutral")
    print(f"\nReddit Sentiment: {result.ticker}")
    print(f"  Score: {result.score:+.2f} ({label})")
    print(f"  Sample: {result.sample_size} posts")
    if result.top_posts:
        print("\n  Top Posts:")
        for i, p in enumerate(result.top_posts, 1):
            print(f"    {i}. [{p.subreddit}] {p.title} (score={p.score}, ratio={p.upvote_ratio:.0%})")


def _cmd_macro(args: list[str]) -> None:
    """Fetch World Bank macroeconomic indicators."""
    import argparse

    from quant_tools.data.providers.worldbank import INDICATORS

    parser = argparse.ArgumentParser(
        prog="quant-tools macro",
        description="Fetch World Bank macroeconomic indicators.",
    )
    parser.add_argument(
        "country", help="ISO country code (e.g. US, GB, CN)",
    )
    parser.add_argument(
        "--indicator", default=None,
        choices=list(INDICATORS.keys()),
        help="Specific indicator (default: all)",
    )
    parser.add_argument(
        "--range", default="2000:2025", dest="date_range",
        help="Year range (default: 2000:2025)",
    )
    parsed = parser.parse_args(args)

    from quant_tools.data.providers.worldbank import WorldBankClient

    client = WorldBankClient()
    if parsed.indicator:
        df = client.fetch_indicator(parsed.country, parsed.indicator, parsed.date_range)
        print(f"\n{df['indicator'].iloc[0]} — {df['country'].iloc[0]}")
        print(df[["year", "value"]].to_string(index=False))
    else:
        df = client.fetch_multiple_indicators(parsed.country, date_range=parsed.date_range)
        print(f"\nWorld Bank Macro — {parsed.country.upper()}")
        print(df.to_string(index=False))


def _cmd_news(args: list[str]) -> None:
    """Fetch GDELT news events."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools news",
        description="Search GDELT for news events related to a query.",
    )
    parser.add_argument("query", help="Search query (e.g. 'AAPL earnings')")
    parser.add_argument("--days", type=int, default=30, help="Lookback days (default: 30)")
    parsed = parser.parse_args(args)

    from datetime import date, timedelta

    from quant_tools.data.alternative.gdelt import GdeltClient

    # fetch_events takes a date window and returns a DataFrame; it never had a
    # `days` argument and never returned a list of dicts.
    end = date.today()
    start = end - timedelta(days=parsed.days)
    client = GdeltClient()
    events = client.fetch_events(
        parsed.query,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    print(f"\nGDELT News Events for '{parsed.query}' (last {parsed.days} days):")
    if events.empty:
        print("  No events found.")
        return
    for i, (_, row) in enumerate(events.head(20).iterrows(), 1):
        when = row.get("seendate", row.get("date", "N/A"))
        what = row.get("title") or row.get("url") or "N/A"
        print(f"  {i}. [{when}] {what}")
    print(f"\n  {len(events)} events total.")


def _cmd_trends(args: list[str]) -> None:
    """Fetch Google Trends data."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools trends",
        description="Fetch Google Trends interest over time.",
    )
    parser.add_argument("keywords", nargs="+", help="Keywords to search")
    parser.add_argument("--timeframe", default="today 3-m", help="Timeframe (default: 'today 3-m')")
    parsed = parser.parse_args(args)

    import pandas as pd

    from quant_tools.data.alternative.google_trends import GoogleTrendsClient

    # The client fetches one keyword at a time; there is no
    # fetch_interest_over_time taking a list. Join the per-keyword frames.
    client = GoogleTrendsClient()
    frames = []
    for keyword in parsed.keywords:
        df = client.fetch_interest(keyword, timeframe=parsed.timeframe)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("No trends data available.")
        return

    combined = pd.concat(frames, axis=1)
    print(f"\nGoogle Trends: {', '.join(parsed.keywords)} ({parsed.timeframe})")
    print(combined.tail(10).to_string())


def _cmd_defi(args: list[str]) -> None:
    """Fetch DeFi protocol data from DeFi Llama."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools defi",
        description="Fetch DeFi protocol TVL and metrics from DeFi Llama.",
    )
    parser.add_argument("--protocol", default=None, help="Protocol slug (e.g. 'aave'), shows its TVL history")
    parser.add_argument("--chain", default=None, help="Filter yield pools by chain (e.g. 'Ethereum')")
    parser.add_argument("--top", type=int, default=10, help="Number of top yield pools (default: 10)")
    parsed = parser.parse_args(args)

    # The client is DefiLlamaClient, and it offers TVL history per protocol and
    # a yield-pool table. There is no top-protocols-by-TVL endpoint behind it,
    # so --top ranks the yield pools it can actually fetch.
    from quant_tools.data.providers.onchain import DefiLlamaClient

    client = DefiLlamaClient()
    if parsed.protocol:
        df = client.fetch_tvl(parsed.protocol)
        print(f"\nDeFi Llama TVL history: {parsed.protocol}")
        if df.empty:
            print("  No data found.")
            return
        print(df.tail(10).to_string(index=False))
        latest = df.iloc[-1]
        print(f"\n  Latest TVL: ${float(latest.get('tvl', 0)):,.0f}")
        return

    pools = client.fetch_yields(chain=parsed.chain)
    if pools.empty:
        print("No yield pool data available.")
        return
    top = pools.sort_values("tvl", ascending=False).head(parsed.top)
    scope = f" on {parsed.chain}" if parsed.chain else ""
    print(f"\nTop {parsed.top} DeFi yield pools by TVL{scope}:")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"  {i:2}. {row['project']} / {row['pool']} ({row['chain']}): "
            f"${float(row['tvl']):,.0f} at {float(row['apy']):.2f}% APY"
        )


def _cmd_fear_greed(args: list[str]) -> None:
    """Fetch Fear & Greed Index."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools fear-greed",
        description="Fetch the current Fear & Greed Index.",
    )
    parser.add_argument("--days", type=int, default=7, help="Crypto index history days (default: 7)")
    parser.add_argument(
        "--market", choices=["equity", "crypto"], default="crypto",
        help="CNN's equity index or the crypto index (default: crypto)",
    )
    parsed = parser.parse_args(args)

    # There is no fetch_index. The client exposes CNN's equity gauge as a dict
    # and the crypto gauge as a history frame, so they are printed separately.
    from quant_tools.data.alternative.fear_greed import FearGreedClient

    client = FearGreedClient()
    if parsed.market == "equity":
        data = client.fetch_cnn_index()
        print("\nCNN Fear & Greed Index (equities):")
        for key, value in data.items():
            print(f"  {key}: {value}")
        return

    df = client.fetch_crypto_index(limit=parsed.days)
    print(f"\nCrypto Fear & Greed Index (last {parsed.days} days):")
    if df.empty:
        print("  No data available.")
        return
    print(df.to_string(index=False))


def _cmd_options(args: list[str]) -> None:
    """Fetch options chain data."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools options",
        description="Fetch options chain for a ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--expiry", default=None, help="Expiration date (YYYY-MM-DD)")
    parsed = parser.parse_args(args)

    from quant_tools.data.providers.options import OptionsProvider

    provider = OptionsProvider()
    # The keyword is `expiration`, not `expiry`.
    chain = provider.fetch_chain(parsed.ticker, expiration=parsed.expiry)
    print(f"\nOptions Chain: {chain.symbol}  expiration {chain.expiration}")
    print(f"  Calls: {len(chain.calls)} contracts")
    print(f"  Puts:  {len(chain.puts)} contracts")
    print(f"  Put/call ratio: {provider.calculate_put_call_ratio(chain):.3f}")
    print(f"  Max pain:       {provider.calculate_max_pain(chain):.2f}")

    unusual = provider.scan_unusual_volume(chain)
    if unusual:
        print(f"\n  Unusual volume ({len(unusual)} contracts):")
        for item in unusual[:10]:
            print(f"    {item}")


def _cmd_benchmark_models(args: list[str]) -> None:
    """Benchmark prediction models."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools benchmark-models",
        description="Run benchmark suite on prediction models.",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["AAPL"],
        help="Symbols to benchmark (default: AAPL)",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days (default: 5)")
    parsed = parser.parse_args(args)

    from quant_tools.predict.benchmark import ModelBenchmark

    # run() takes a list of symbols and returns a DataFrame; the old call passed
    # `symbol=` and a `model_names=` argument that does not exist, then treated
    # the result as a dict of dicts.
    bench = ModelBenchmark()
    results = bench.run(parsed.symbols, horizon_days=parsed.horizon)

    print(f"\nModel Benchmark Results ({', '.join(parsed.symbols)}, {parsed.horizon}d horizon)")
    if results.empty:
        print("  No results. Install the ML extra: pip install 'quant-tools[ml]'")
        return
    print(results.to_string(index=False))


def _cmd_risk(args: list[str]) -> None:
    """Calculate portfolio risk metrics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools risk",
        description="Calculate VaR, CVaR, and other risk metrics for a portfolio.",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight (e.g. '{\"AAPL\":0.5,\"BND\":0.5}')",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="VaR confidence (default: 0.95)")
    parser.add_argument("--period", default="1y", help="History window (default: 1y)")
    parsed = parser.parse_args(args)

    holdings = _parse_holdings(parsed.holdings)
    # The class is EnhancedRiskAnalyzer, and it works on a returns array rather
    # than on a holdings dict. There has never been a compute_risk method.
    from quant_tools.portfolio.risk_enhanced import EnhancedRiskAnalyzer

    returns = _portfolio_returns(holdings, period=parsed.period)
    analyzer = EnhancedRiskAnalyzer()
    var = analyzer.var_trio(returns, confidence=parsed.confidence)
    tail = analyzer.tail_metrics(returns)

    print(f"\nRisk Metrics ({parsed.confidence:.0%} confidence, {parsed.period} of daily returns)")
    print("  Value at Risk")
    print(f"    parametric:  {var.parametric:>9.2%}")
    print(f"    historical:  {var.historical:>9.2%}")
    print(f"    monte carlo: {var.monte_carlo:>9.2%}")
    print("  Tail risk")
    print(f"    EVaR:              {tail.evar:>9.2%}")
    print(f"    CDaR:              {tail.cdar:>9.2%}")
    print(f"    max DD duration:   {tail.max_dd_duration:>6} days")
    print(f"    tail ratio:        {tail.tail_ratio:>9.3f}")


def _cmd_stress_test(args: list[str]) -> None:
    """Run portfolio stress tests."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools stress-test",
        description="Run historical stress test scenarios on a portfolio.",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight",
    )
    parser.add_argument(
        "--scenario", default=None,
        help="One scenario to run (default: every historical scenario)",
    )
    parser.add_argument("--period", default="1y", help="History window (default: 1y)")
    parsed = parser.parse_args(args)

    holdings = _parse_holdings(parsed.holdings)
    from quant_tools.portfolio.risk_enhanced import (
        HISTORICAL_SCENARIOS,
        EnhancedRiskAnalyzer,
    )

    # stress_test runs one named scenario at a time against a returns array, so
    # loop the historical set rather than expecting it to return a whole table.
    returns = _portfolio_returns(holdings, period=parsed.period)
    analyzer = EnhancedRiskAnalyzer()
    names = [parsed.scenario] if parsed.scenario else list(HISTORICAL_SCENARIOS)

    print("\nStress Test Results:")
    for name in names:
        result = analyzer.stress_test(returns, holdings, name)
        loss = result.get("portfolio_loss")
        loss_text = f"{loss:+.2%}" if isinstance(loss, float) else str(loss)
        print(f"\n  {result.get('scenario', name)}: {loss_text}")
        description = result.get("description")
        if description:
            print(f"    {description}")


def _cmd_regime(args: list[str]) -> None:
    """Detect market regime using HMM."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools regime",
        description="Detect current market regime using Hidden Markov Model.",
    )
    parser.add_argument("--ticker", default="SPY", help="Index/ETF to analyze (default: SPY)")
    parser.add_argument("--period", default="5y", help="History window (default: 5y)")
    parser.add_argument("--regimes", type=int, default=3, help="Number of regimes (default: 3)")
    parsed = parser.parse_args(args)

    import numpy as np

    from quant_tools.data.market_data import fetch_prices
    from quant_tools.portfolio.regime import RegimeDetector

    # The detector fits on a returns array; there is no detect(ticker). Fetch the
    # prices here and hand it what it actually wants.
    df = fetch_prices(parsed.ticker, period=parsed.period)
    close = df["Close"].values.astype(float)
    returns = np.diff(close) / close[:-1]

    detector = RegimeDetector(n_regimes=parsed.regimes)
    detector.fit(returns)
    state = detector.get_current_state(returns)

    print(f"\nMarket Regime Detection ({parsed.ticker}, {parsed.period})")
    print(f"  Current regime: {state.label} (id {state.regime_id})")
    print(f"  Confidence:     {state.probability:.1%}")

    if state.transition_probs:
        # transition_probs is already keyed by regime label, not indexed by id.
        print("  Probability of moving to:")
        for label, prob in sorted(state.transition_probs.items(), key=lambda kv: -kv[1]):
            print(f"    {label:<12} {prob:.1%}")

    allocation = detector.suggest_allocation(state.label)
    if allocation:
        print("  Suggested allocation:")
        for asset, weight in allocation.items():
            print(f"    {asset:<12} {weight:.0%}")


def _cmd_analyze(args: list[str]) -> None:
    """AI market analysis via LLM."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools analyze",
        description="AI-powered market analysis for a ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--depth", choices=["quick", "deep"], default="quick",
        help="quick reads 6 months of bars, deep reads 2 years",
    )
    parsed = parser.parse_args(args)

    from quant_tools.data.market_data import fetch_prices
    from quant_tools.llm.analysis import MarketAnalyzer

    # analyze() reads the bars you give it and returns prose. It never took a
    # `depth` argument, so depth now selects how much history to hand over.
    period = "6mo" if parsed.depth == "quick" else "2y"
    ohlcv = fetch_prices(parsed.ticker, period=period)

    analyzer = MarketAnalyzer()
    result = analyzer.analyze(parsed.ticker, ohlcv)
    print(f"\nAI Market Analysis: {parsed.ticker} ({parsed.depth}, {period} of bars)")
    print("=" * 60)
    print(result)


def _cmd_ask(args: list[str]) -> None:
    """Ask natural language questions about portfolio/market."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools ask",
        description="Ask natural language questions about your portfolio or market.",
    )
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight — the portfolio the question is about",
    )
    parsed = parser.parse_args(args)

    question = " ".join(parsed.question)
    holdings = _parse_holdings(parsed.holdings)
    from quant_tools.llm.query import PortfolioQueryEngine

    # The method is query(question, portfolio), not ask(question). It cannot
    # answer a portfolio question without being told the portfolio, which is why
    # --holdings is now required rather than silently absent.
    engine = PortfolioQueryEngine()
    answer = engine.query(question, holdings)
    print(f"\nQ: {question}")
    print(f"\nA: {answer}")


def _cmd_tearsheet(args: list[str]) -> None:
    """Generate portfolio tearsheet report."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="quant-tools tearsheet",
        description="Generate a portfolio tearsheet as a standalone HTML file.",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight",
    )
    parser.add_argument("--output", default="tearsheet.html", help="Output file (default: tearsheet.html)")
    parser.add_argument("--period", default="1y", help="History window (default: 1y)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    parsed = parser.parse_args(args)

    holdings = _parse_holdings(parsed.holdings)

    import pandas as pd

    from quant_tools.data.market_data import fetch_prices
    from quant_tools.reporting.tearsheet import TearsheetGenerator, compute_metrics

    # The generator renders HTML from a TearsheetData; it has no generate() and
    # produces no PDF, so the command now writes the HTML it really makes and
    # the default filename says so.
    returns = pd.Series(_portfolio_returns(holdings, period=parsed.period))
    benchmark_prices = fetch_prices(parsed.benchmark, period=parsed.period)
    benchmark = benchmark_prices["Close"].pct_change().dropna().reset_index(drop=True)

    data = compute_metrics(returns, benchmark_returns=benchmark)
    html = TearsheetGenerator(title=f"Tearsheet: {', '.join(holdings)}").generate_html(data)

    output = Path(parsed.output)
    if output.parent != Path(""):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"\nTearsheet written to {output.resolve()}")
    print(f"  Total return: {data.total_return:.2%}   CAGR: {data.cagr:.2%}")
    print(f"  Sharpe: {data.sharpe_ratio:.2f}   Max drawdown: {data.max_drawdown:.2%}")


def _cmd_paper_trade(args: list[str]) -> None:
    """Execute paper trades via Alpaca or crypto simulation."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools paper-trade",
        description="Submit paper trades via Alpaca (equity) or crypto simulator.",
    )
    parser.add_argument("action", choices=["buy", "sell", "status", "positions"], help="Trade action")
    parser.add_argument("--symbol", default=None, help="Symbol to trade")
    parser.add_argument("--qty", type=float, default=None, help="Quantity")
    parser.add_argument("--provider", choices=["alpaca", "crypto"], default="alpaca", help="Broker")
    parsed = parser.parse_args(args)

    if parsed.action in ("buy", "sell") and (not parsed.symbol or not parsed.qty):
        raise DataError("--symbol and --qty are required for buy and sell")

    if parsed.provider == "crypto":
        # crypto_paper exports CryptoPaperEngine, an in-memory simulator with no
        # broker behind it. It starts empty every run, so it can price and fill an
        # order but has no account history to report.
        from quant_tools.execution.crypto_paper import CryptoPaperEngine, Side

        if parsed.action in ("status", "positions"):
            raise DataError(
                "The crypto simulator holds no state between runs, so it has no "
                "account or positions to show. Use --provider alpaca for those, "
                "or place a simulated order with buy/sell."
            )

        from quant_tools.data.market_data import fetch_prices

        price = float(fetch_prices(parsed.symbol, period="5d")["Close"].iloc[-1])
        engine = CryptoPaperEngine()
        engine.update_price(parsed.symbol, price)
        order = engine.submit_order(
            parsed.symbol,
            Side.BUY if parsed.action == "buy" else Side.SELL,
            parsed.qty,
            price,
        )
        print(f"\nSimulated {parsed.action}: {order}")
        print(f"  Equity after fill: ${engine.equity():,.2f}")
        return

    # AlpacaPaperTrader exposes get_account/get_positions/submit_order. There has
    # never been a get_account_status.
    from quant_tools.execution.alpaca import AlpacaPaperTrader, OrderSide

    trader = AlpacaPaperTrader()
    if parsed.action == "status":
        account = trader.get_account()
        if account is None:
            raise DataError(
                "No Alpaca account returned. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in .env, and install the realtime extra."
            )
        print("\nAlpaca Paper Account:")
        print(f"  Equity:          ${account.equity:,.2f}")
        print(f"  Cash:            ${account.cash:,.2f}")
        print(f"  Buying power:    ${account.buying_power:,.2f}")
        print(f"  Portfolio value: ${account.portfolio_value:,.2f}")
    elif parsed.action == "positions":
        positions = trader.get_positions()
        print("\nOpen Positions:")
        if not positions:
            print("  none")
        for pos in positions:
            print(
                f"  {pos.symbol:<8} qty={pos.qty:<10.4f} entry={pos.avg_entry:<10.2f} "
                f"last={pos.current_price:<10.2f} pnl={pos.unrealized_pnl:+,.2f}"
            )
    else:
        result = trader.submit_order(
            parsed.symbol,
            parsed.qty,
            side=OrderSide.BUY if parsed.action == "buy" else OrderSide.SELL,
        )
        print(f"\nOrder submitted: {result}")


def _cmd_dashboard(args: list[str]) -> None:
    """Launch the TUI dashboard."""
    # Every other command routes --help through argparse. This one took no
    # options, so it went straight into the full-screen TUI and left anyone who
    # typed `quant-tools dashboard --help` staring at a terminal they then had
    # to quit out of.
    if _wants_help(args):
        print("usage: quant-tools dashboard")
        print()
        print("Launch the terminal dashboard. Takes no options; press q to quit.")
        print("For the browser version, use: quant-tools web")
        return

    from quant_tools.tui.app import run_dashboard
    run_dashboard()


def _cmd_web(args: list[str]) -> None:
    """Launch the Streamlit web dashboard.

    Args:
        args: Extra streamlit options, e.g. ["--server.port", "8599"]. Anything
            given here replaces the matching default below.
    """
    import subprocess
    from pathlib import Path

    # Without this, --help was handed to `streamlit run`, which answered with
    # streamlit's own help. On Windows that help text is not even printable: it
    # contains an arrow glyph that cp1252 cannot encode, so the command died on
    # a UnicodeEncodeError from inside click.
    if _wants_help(args):
        print("usage: quant-tools web [streamlit options]")
        print()
        print("Launch the Streamlit dashboard, by default on http://localhost:8501")
        print("with the dark theme applied.")
        print()
        print("Any streamlit option given here replaces the matching default,")
        print("for example:")
        print("  quant-tools web --server.port 8599")
        print("  quant-tools web --server.headless true")
        return

    app = Path(__file__).parent / "dashboard" / "app.py"
    # Streamlit resolves .streamlit/config.toml against the working directory,
    # not the script, so an installed copy launched from anywhere else picked up
    # no theme at all: light chrome around this app's dark cards, with headings
    # rendering dark on dark. Pass the theme explicitly so it holds everywhere.
    # Values match dashboard/core/colors.py and .streamlit/config.toml.
    defaults = {
        "--server.headless": "false",
        "--browser.gatherUsageStats": "false",
        "--client.toolbarMode": "minimal",
        "--theme.base": "dark",
        "--theme.primaryColor": "#3b82f6",
        "--theme.backgroundColor": "#0e1117",
        "--theme.secondaryBackgroundColor": "#1c2128",
        "--theme.textColor": "#f8fafc",
    }

    # args used to be dropped on the floor, so the documented
    # "quant-tools web --server.port 8599" silently started on 8501, and
    # --server.headless could not be turned on at all. Drop any default the
    # caller has overridden rather than passing the same flag twice.
    overridden = {a.split("=", 1)[0] for a in args if a.startswith("--")}
    flags: list[str] = []
    for flag, value in defaults.items():
        if flag not in overridden:
            flags += [flag, value]

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), *flags, *args],
    )


def _print_help() -> None:
    """Print available commands."""
    print("quant-tools - Quantitative Finance Toolkit")
    print()
    print("Data & Analysis:")
    print("  scan             Scan universe for signals")
    print("  filings          Download SEC EDGAR filings")
    print("  sentiment        Reddit sentiment for a ticker")
    print("  macro            World Bank macroeconomic data")
    print("  news             GDELT news event search")
    print("  trends           Google Trends data")
    print("  defi             DeFi Llama protocol data")
    print("  fear-greed       Fear & Greed Index")
    print("  options          Options chain data")
    print()
    print("Prediction & Backtesting:")
    print("  predict          ML ensemble prediction for any asset")
    print("  track            View stored predictions + model performance")
    print("  resolve          Resolve pending predictions vs actuals")
    print("  clear            Clear stored predictions")
    print("  benchmark-models Benchmark prediction model suite")
    print("  backtest         Run strategy backtest")
    print()
    print("Portfolio & Risk:")
    print("  portfolio        Analyze portfolio")
    print("  factors          Factor scoring + macro regime")
    print("  risk             Calculate VaR, CVaR, risk metrics")
    print("  stress-test      Run portfolio stress tests")
    print("  regime           Detect market regime (HMM)")
    print()
    print("AI & LLM:")
    print("  analyze          AI-powered market analysis")
    print("  ask              Natural language portfolio queries")
    print()
    print("Reporting & Execution:")
    print("  tearsheet        Generate portfolio tearsheet (PDF)")
    print("  paper-trade      Paper trading (Alpaca / crypto sim)")
    print()
    print("Dashboard:")
    print("  dashboard        Launch TUI dashboard")
    print("  web              Launch Streamlit web dashboard")
    print("  help             Show this help")
    print()
    print("Examples:")
    print("  quant-tools scan --universe sp500 --top-n 20")
    print("  quant-tools backtest --ticker AAPL --strategy sma_crossover")
    print("  quant-tools portfolio --holdings '{\"AAPL\":0.3,\"BND\":0.7}'")
    print("  quant-tools predict AAPL --horizon 10")
    print("  quant-tools news 'AAPL earnings'")
    print("  quant-tools trends AAPL MSFT --timeframe 'today 3-m'")
    print("  quant-tools risk --holdings '{\"AAPL\":0.5,\"BND\":0.5}'")
    print("  quant-tools analyze AAPL --depth deep")
    print("  quant-tools ask 'What is my portfolio beta?'")
    print("  quant-tools tearsheet --holdings '{\"AAPL\":0.3,\"BND\":0.7}'")
    print("  quant-tools paper-trade buy --symbol AAPL --qty 10")
    print("  quant-tools regime --ticker SPY")


if __name__ == "__main__":
    main()
