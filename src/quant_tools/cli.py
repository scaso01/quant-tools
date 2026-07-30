"""Unified CLI dispatcher: quant-tools <command>."""

from __future__ import annotations

import sys

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

    rc = handler(args)
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)


def _cmd_scan(args: list[str]) -> None:
    """Run the market scanner."""
    from quant_tools.scanner.cli import main as scan_main
    scan_main(args)


def _cmd_backtest(args: list[str]) -> None:
    """Run a backtest."""
    from quant_tools.backtest.cli import backtest_command
    backtest_command(args)


def _cmd_portfolio(args: list[str]) -> None:
    """Run portfolio analysis."""
    from quant_tools.portfolio.cli import main as portfolio_main
    portfolio_main(args)


def _cmd_factors(args: list[str]) -> None:
    """Run factor analysis."""
    from quant_tools.factors.cli import main as factors_main
    factors_main(args)


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
        print(f"\n  Top Posts:")
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

    from quant_tools.data.alternative.gdelt import GdeltClient

    client = GdeltClient()
    events = client.fetch_events(parsed.query, days=parsed.days)
    print(f"\nGDELT News Events for '{parsed.query}' (last {parsed.days} days):")
    for i, ev in enumerate(events[:20], 1):
        print(f"  {i}. [{ev.get('date', 'N/A')}] {ev.get('title', ev.get('url', 'N/A'))}")
    if not events:
        print("  No events found.")


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

    from quant_tools.data.alternative.google_trends import GoogleTrendsClient

    client = GoogleTrendsClient()
    df = client.fetch_interest_over_time(parsed.keywords, timeframe=parsed.timeframe)
    if df is not None and not df.empty:
        print(f"\nGoogle Trends: {', '.join(parsed.keywords)}")
        print(df.tail(10).to_string())
    else:
        print("No trends data available.")


def _cmd_defi(args: list[str]) -> None:
    """Fetch DeFi protocol data from DeFi Llama."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools defi",
        description="Fetch DeFi protocol TVL and metrics from DeFi Llama.",
    )
    parser.add_argument("--protocol", default=None, help="Specific protocol slug (e.g. 'aave')")
    parser.add_argument("--top", type=int, default=10, help="Number of top protocols (default: 10)")
    parsed = parser.parse_args(args)

    from quant_tools.data.providers.onchain import OnChainClient

    client = OnChainClient()
    if parsed.protocol:
        data = client.fetch_protocol_tvl(parsed.protocol)
        print(f"\nDeFi Llama: {parsed.protocol}")
        if data:
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print("  No data found.")
    else:
        protocols = client.fetch_top_protocols(limit=parsed.top)
        print(f"\nTop {parsed.top} DeFi Protocols by TVL:")
        for i, p in enumerate(protocols[:parsed.top], 1):
            name = p.get("name", "Unknown")
            tvl = p.get("tvl", 0)
            print(f"  {i}. {name}: ${tvl:,.0f}")


def _cmd_fear_greed(args: list[str]) -> None:
    """Fetch Fear & Greed Index."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools fear-greed",
        description="Fetch the current Fear & Greed Index.",
    )
    parser.add_argument("--days", type=int, default=7, help="History days (default: 7)")
    parsed = parser.parse_args(args)

    from quant_tools.data.alternative.fear_greed import FearGreedClient

    client = FearGreedClient()
    data = client.fetch_index(days=parsed.days)
    print("\nFear & Greed Index:")
    if isinstance(data, list):
        for entry in data[:parsed.days]:
            val = entry.get("value", "N/A")
            label = entry.get("value_classification", "N/A")
            ts = entry.get("timestamp", "N/A")
            print(f"  [{ts}] {val} ({label})")
    elif isinstance(data, dict):
        val = data.get("value", "N/A")
        label = data.get("value_classification", "N/A")
        print(f"  Current: {val} ({label})")
    else:
        print(f"  {data}")


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
    chain = provider.fetch_chain(parsed.ticker, expiry=parsed.expiry)
    if chain is not None:
        print(f"\nOptions Chain: {parsed.ticker}")
        if hasattr(chain, "calls") and chain.calls is not None:
            print(f"  Calls: {len(chain.calls)} contracts")
        if hasattr(chain, "puts") and chain.puts is not None:
            print(f"  Puts: {len(chain.puts)} contracts")
        if hasattr(chain, "summary"):
            for k, v in chain.summary.items():
                print(f"  {k}: {v}")
    else:
        print(f"No options data for {parsed.ticker}")


def _cmd_benchmark_models(args: list[str]) -> None:
    """Benchmark prediction models."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools benchmark-models",
        description="Run benchmark suite on prediction models.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Symbol to benchmark (default: AAPL)")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to benchmark (default: all)")
    parsed = parser.parse_args(args)

    from quant_tools.predict.benchmark import ModelBenchmark

    bench = ModelBenchmark()
    results = bench.run(symbol=parsed.symbol, model_names=parsed.models)
    print(f"\nModel Benchmark Results ({parsed.symbol}):")
    for name, metrics in results.items():
        print(f"\n  {name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


def _cmd_risk(args: list[str]) -> None:
    """Calculate portfolio risk metrics."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="quant-tools risk",
        description="Calculate VaR, CVaR, and other risk metrics for a portfolio.",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight (e.g. '{\"AAPL\":0.5,\"BND\":0.5}')",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="VaR confidence (default: 0.95)")
    parsed = parser.parse_args(args)

    holdings = json.loads(parsed.holdings)
    from quant_tools.portfolio.risk_enhanced import RiskAnalyzer

    analyzer = RiskAnalyzer()
    metrics = analyzer.compute_risk(holdings, confidence=parsed.confidence)
    print(f"\nRisk Metrics ({parsed.confidence:.0%} confidence):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def _cmd_stress_test(args: list[str]) -> None:
    """Run portfolio stress tests."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="quant-tools stress-test",
        description="Run historical stress test scenarios on a portfolio.",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight",
    )
    parsed = parser.parse_args(args)

    holdings = json.loads(parsed.holdings)
    from quant_tools.portfolio.risk_enhanced import RiskAnalyzer

    analyzer = RiskAnalyzer()
    results = analyzer.stress_test(holdings)
    print("\nStress Test Results:")
    for scenario, impact in results.items():
        if isinstance(impact, float):
            print(f"  {scenario}: {impact:+.2%}")
        else:
            print(f"  {scenario}: {impact}")


def _cmd_regime(args: list[str]) -> None:
    """Detect market regime using HMM."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools regime",
        description="Detect current market regime using Hidden Markov Model.",
    )
    parser.add_argument("--ticker", default="SPY", help="Index/ETF to analyze (default: SPY)")
    parsed = parser.parse_args(args)

    from quant_tools.portfolio.regime import RegimeDetector

    detector = RegimeDetector()
    regime = detector.detect(parsed.ticker)
    print(f"\nMarket Regime Detection ({parsed.ticker}):")
    if isinstance(regime, dict):
        for k, v in regime.items():
            print(f"  {k}: {v}")
    else:
        print(f"  Current regime: {regime}")


def _cmd_analyze(args: list[str]) -> None:
    """AI market analysis via LLM."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools analyze",
        description="AI-powered market analysis for a ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--depth", choices=["quick", "deep"], default="quick", help="Analysis depth")
    parsed = parser.parse_args(args)

    from quant_tools.llm.analysis import MarketAnalyzer

    analyzer = MarketAnalyzer()
    result = analyzer.analyze(parsed.ticker, depth=parsed.depth)
    print(f"\nAI Market Analysis: {parsed.ticker}")
    print(f"{'='*60}")
    if isinstance(result, dict):
        for section, content in result.items():
            print(f"\n{section}:")
            print(f"  {content}")
    else:
        print(result)


def _cmd_ask(args: list[str]) -> None:
    """Ask natural language questions about portfolio/market."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="quant-tools ask",
        description="Ask natural language questions about your portfolio or market.",
    )
    parser.add_argument("question", nargs="+", help="Your question")
    parsed = parser.parse_args(args)

    question = " ".join(parsed.question)
    from quant_tools.llm.query import PortfolioQueryEngine

    engine = PortfolioQueryEngine()
    answer = engine.ask(question)
    print(f"\nQ: {question}")
    print(f"\nA: {answer}")


def _cmd_tearsheet(args: list[str]) -> None:
    """Generate portfolio tearsheet report."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="quant-tools tearsheet",
        description="Generate a comprehensive portfolio tearsheet (PDF).",
    )
    parser.add_argument(
        "--holdings", required=True, type=str,
        help="JSON dict of symbol:weight",
    )
    parser.add_argument("--output", default="tearsheet.pdf", help="Output file (default: tearsheet.pdf)")
    parsed = parser.parse_args(args)

    holdings = json.loads(parsed.holdings)
    from quant_tools.reporting.tearsheet import TearsheetGenerator

    gen = TearsheetGenerator()
    path = gen.generate(holdings, output_path=parsed.output)
    print(f"\nTearsheet generated: {path}")


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

    if parsed.provider == "alpaca":
        from quant_tools.execution.alpaca import AlpacaPaperTrader
        trader = AlpacaPaperTrader()
    else:
        from quant_tools.execution.crypto_paper import CryptoPaperTrader
        trader = CryptoPaperTrader()

    if parsed.action == "status":
        status = trader.get_account_status()
        print("\nAccount Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
    elif parsed.action == "positions":
        positions = trader.get_positions()
        print("\nOpen Positions:")
        for pos in positions:
            print(f"  {pos}")
    elif parsed.action in ("buy", "sell"):
        if not parsed.symbol or not parsed.qty:
            print("Error: --symbol and --qty required for buy/sell")
            sys.exit(1)
        result = trader.submit_order(parsed.symbol, parsed.qty, parsed.action)
        print(f"\nOrder submitted: {result}")


def _cmd_dashboard(args: list[str]) -> None:
    """Launch the TUI dashboard."""
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
