"""Per-page help system with glossary and workflow guidance."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Glossary ─────────────────────────────────────────────────────────

GLOSSARY: dict[str, str] = {
    "Alpha": "Excess return above a benchmark. Positive alpha means you outperformed the market.",
    "Beta": "How much an asset moves relative to the market. Beta > 1 means more volatile than the market.",
    "CAGR": "Compound Annual Growth Rate — the smoothed annual return over a period.",
    "Confidence": "How certain a prediction model is about its forecast, from 0% to 100%.",
    "Consensus Signal": "The combined recommendation when multiple models agree on a direction.",
    "CVaR": "Conditional Value at Risk — the average loss in the worst-case scenarios beyond VaR.",
    "Drawdown": "The peak-to-trough decline in portfolio value. Max drawdown is the worst drop.",
    "Efficient Frontier": "The set of portfolios offering the highest return for each level of risk.",
    "Ensemble": "A group of models that vote together — usually more accurate than any single model.",
    "Factor Z-Score": "How many standard deviations a stock's factor score is from the average.",
    "GARCH": "A volatility model that captures how market turbulence clusters over time.",
    "GDP": "Gross Domestic Product — the total value of goods and services produced by a country.",
    "Gradient Boost": "A machine learning model that builds predictions by combining many small decision trees.",
    "HHI": "Herfindahl-Hirschman Index — measures portfolio concentration. Lower = more diversified.",
    "Information Ratio": "Risk-adjusted excess return vs. a benchmark, per unit of tracking error.",
    "LLM": "Large Language Model — an AI that generates text analysis from prompts.",
    "Macro Regime": "The current phase of the economic cycle: Expansion, Slowdown, Contraction, or Recovery.",
    "Markowitz": "Modern Portfolio Theory optimization that maximizes return for a given risk level.",
    "Max Drawdown": "The largest peak-to-trough decline during a period — measures worst-case loss.",
    "Mean Reversion": "A signal that suggests a stock has moved too far from its average and may snap back.",
    "Momentum": "A signal based on recent price trends — stocks moving up tend to keep moving up.",
    "Monte Carlo": "A simulation running thousands of random scenarios to estimate future outcomes.",
    "Profit Factor": "Total profits divided by total losses. Above 1.0 means the strategy is profitable.",
    "Quality": "A signal based on financial health — high margins, low debt, strong earnings growth.",
    "Risk Parity": "Portfolio allocation where each asset contributes equal risk, not equal weight.",
    "Rolling Sharpe": "Sharpe ratio calculated over a sliding window to see how it changes over time.",
    "RSI": "Relative Strength Index — measures if a stock is overbought (>70) or oversold (<30).",
    "Sharpe Ratio": "Return per unit of risk. Higher is better. Above 1.0 is generally considered good.",
    "Signal Score": "A composite ranking from 0-100 based on multiple technical indicators.",
    "Sortino Ratio": "Like Sharpe but only penalizes downside volatility — ignores upside swings.",
    "Tracking Error": "How closely a portfolio follows its benchmark. Lower = more closely tracked.",
    "Treynor Ratio": "Return per unit of market risk (beta). Useful for comparing diversified portfolios.",
    "Universe": "The pool of assets being analyzed — e.g., S&P 500, Russell 1000, or a custom list.",
    "Value": "A signal based on valuation ratios — low P/E, low P/B stocks that may be underpriced.",
    "VaR": "Value at Risk — the maximum expected loss at a given confidence level (e.g., 95%).",
    "VWAP": "Volume-Weighted Average Price — the average price weighted by trading volume.",
    "Win Rate": "Percentage of predictions that were correct. 60%+ is considered strong.",
    "Yield Spread": "Difference between long-term and short-term bond yields — signals economic outlook.",
    # ── Backtest Terms ───────────────────────────────────────────────
    "Equity Curve": "A chart showing how portfolio value changes over time during a backtest.",
    "Commission": "Trading fees deducted each time you buy or sell. Reduces net returns.",
    "SMA": "Simple Moving Average — the mean closing price over a fixed window (e.g., 50 days).",
    "Bollinger Bands": "A volatility envelope: SMA plus/minus 2 standard deviations. Price near the bands may signal reversals.",
    "RSI (Indicator)": "Relative Strength Index plotted as a 0-100 oscillator. Used to spot overbought and oversold conditions.",
    "Overbought": "When RSI exceeds 70, the asset may be due for a pullback.",
    "Oversold": "When RSI drops below 30, the asset may be due for a bounce.",
    # ── Portfolio Terms ──────────────────────────────────────────────
    "Correlation": "How closely two assets move together, from -1 (opposite) to +1 (identical).",
    "Sector Exposure": "The percentage of your portfolio allocated to each market sector (Tech, Health, etc.).",
    "Rebalance": "Adjusting your portfolio weights back to target allocations, usually on a schedule.",
    "Benchmark": "A reference index (like S&P 500) used to measure whether your portfolio outperforms the market.",
    # ── Scanner Terms ────────────────────────────────────────────────
    # Signal Score and Universe already exist above
    # ── Factor Terms ─────────────────────────────────────────────────
    "Z-Score": "How many standard deviations a value is from the mean. Higher magnitude means more unusual.",
    "Composite Score": "A single score combining multiple factor signals into one ranked number.",
    "Factor Weight": "How much influence each factor (value, momentum, quality) has in the composite score.",
    "Regime": "The current economic phase (expansion, slowdown, contraction, recovery) detected from macro data.",
    # ── Additional Backtest Terms ──────────────────────────────────────
    "Slippage": "The difference between the expected trade price and the actual execution price.",
    "Walk-Forward": "A validation method that trains on past data and tests on future data, sliding the window forward.",
    "Lookback Period": "The number of historical bars or days an indicator uses to calculate its value.",
    # ── Additional Portfolio Terms ────────────────────────────────────
    "Diversification Ratio": "How much risk is reduced by combining assets. Higher means more diversification benefit.",
    "Tail Risk": "The probability of extreme losses beyond normal expectations — the 'fat tails' of return distributions.",
    "Annualized Volatility": "Standard deviation of returns scaled to one year. Higher means wider price swings.",
    "Weight": "The fraction of total portfolio value allocated to a single asset, expressed as a percentage.",
    # ── Prediction Terms ─────────────────────────────────────────────
    "Horizon": "How many days ahead the prediction targets. Shorter horizons are harder to forecast.",
    "Magnitude": "The predicted size of the price move, not just the direction.",
    "Calibration": "How well a model's stated confidence matches its actual accuracy. Well-calibrated = 80% confident predictions are right 80% of the time.",
}


def jargon(term: str) -> str:
    """Return an HTML <abbr> element with a glossary tooltip for a finance term.

    If the term is not in the glossary, returns the term unchanged.
    Use inside st.markdown(..., unsafe_allow_html=True) contexts.
    """
    definition = GLOSSARY.get(term, "")
    if not definition:
        return term
    # Escape quotes in the definition for the title attribute
    escaped = definition.replace('"', '&quot;')
    return f'<abbr title="{escaped}">{term}</abbr>'


# ── Page Help Data ───────────────────────────────────────────────────


@dataclass
class PageHelp:
    """Help content for a single dashboard page."""

    summary: str
    steps: list[str]
    key_terms: list[str] = field(default_factory=list)
    workflow_position: str = ""
    related_pages: list[tuple[str, str]] = field(default_factory=list)
    status: str = "active"


PAGE_HELP: dict[str, PageHelp] = {
    "scanner": PageHelp(
        summary="Find stocks, crypto, forex, or commodities with strong trading signals.",
        steps=[
            "Pick an **asset class** (equity, crypto, forex, commodity) in the sidebar.",
            "Select a **universe** — the pool of assets to scan (e.g., S&P 500).",
            "Check the **signals** you want to scan for (momentum, value, quality, mean reversion).",
            "Click **Run Scan** and review the ranked results table.",
            "Expand any result to see how its signal score was calculated.",
        ],
        key_terms=["Signal Score", "RSI", "Momentum", "Mean Reversion", "Quality", "Value", "Universe"],
        workflow_position="**Scanner** → Backtester → Portfolio",
        related_pages=[
            ("Strategy Backtester", "Found a stock? Test a strategy on it"),
            ("Predictions", "Get ML predictions for scanned stocks"),
        ],
    ),
    "backtest": PageHelp(
        summary="Test a trading strategy on historical data before risking real money.",
        steps=[
            "Enter a **ticker symbol** (e.g., AAPL) in the sidebar.",
            "Pick a **strategy** — each uses different rules to buy and sell.",
            "Set the **date range** and **initial capital**.",
            "Click **Run Backtest** to see the equity curve and performance metrics.",
            "Try the **Compare Strategies** section to see how strategies stack up.",
        ],
        key_terms=["CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Profit Factor"],
        workflow_position="Scanner → **Backtester** → Portfolio",
        related_pages=[
            ("Portfolio Analyzer", "Strategy working? Build a portfolio around it"),
            ("Market Scanner", "Find more stocks to test"),
        ],
    ),
    "portfolio": PageHelp(
        summary="Analyze a basket of holdings — risk, return, optimization, and Monte Carlo simulation.",
        steps=[
            "Enter your **holdings** as JSON (ticker → weight) or use a preset portfolio.",
            "Set the **time period** and choose a **benchmark** to compare against.",
            "Click **Analyze Portfolio** to view metrics, correlations, and sector exposure.",
            "Scroll down for **Monte Carlo simulation** — thousands of scenarios for your portfolio.",
            "Check the **Efficient Frontier** to see if your allocation is optimal.",
        ],
        key_terms=["Alpha", "Beta", "Sharpe Ratio", "Efficient Frontier", "Monte Carlo", "Risk Parity", "Markowitz", "HHI", "Tracking Error"],
        workflow_position="Scanner → Backtester → **Portfolio** → Risk",
        related_pages=[
            ("Risk Dashboard", "Deep-dive into portfolio risk metrics"),
            ("Factors & Macro", "Check factor exposure of your holdings"),
        ],
    ),
    "factors": PageHelp(
        summary="Score stocks by value, momentum, and quality factors with macroeconomic overlay.",
        steps=[
            "Optionally click **Detect Macro Regime** to check the current economic phase.",
            "Pick a **universe** (S&P 500 top 100 or enter custom tickers).",
            "Adjust **factor weights** — how much to emphasize value vs. momentum vs. quality.",
            "Review the **ranked results** sorted by composite score.",
            "Check the **regime tilts** to see how the macro environment affects recommendations.",
        ],
        key_terms=["Factor Z-Score", "Macro Regime", "GDP", "Yield Spread", "Momentum", "Value", "Quality"],
        workflow_position="**Factors** → Predictions → Performance",
        related_pages=[
            ("Predictions", "Get ML forecasts for top-scoring stocks"),
            ("Portfolio Analyzer", "Analyze your factor-based picks as a portfolio"),
        ],
    ),
    "predictions": PageHelp(
        summary="ML ensemble forecasts — which direction and how confident the models are.",
        steps=[
            "Use the **filters** to narrow by asset class, horizon, or model.",
            "Review the **active predictions** table with confidence levels and consensus.",
            "Check the **Model Agreement** chart to see if models agree or disagree.",
            "Click **New Prediction** to generate forecasts for specific symbols.",
            "Track outcomes on the **Performance Tracker** page.",
        ],
        key_terms=["Ensemble", "Gradient Boost", "GARCH", "Confidence", "Consensus Signal"],
        workflow_position="Factors → **Predictions** → Performance",
        related_pages=[
            ("Performance", "See how well these predictions turned out"),
            ("Market Scanner", "Find new symbols to predict"),
        ],
    ),
    "performance": PageHelp(
        summary="Track whether the prediction models were actually right.",
        steps=[
            "View the **overall win rate** and best/worst performing models.",
            "Check the **model comparison table** for per-model stats.",
            "Review the **heatmap** to see which models work best for which asset classes.",
            "Use the **confidence vs. accuracy** scatter to check calibration.",
            "Click **Resolve** to update prediction outcomes with current prices.",
        ],
        key_terms=["Win Rate", "Confidence", "Profit Factor"],
        workflow_position="Predictions → **Performance**",
        related_pages=[
            ("Predictions", "Generate new predictions to track"),
        ],
    ),
    "rolling": PageHelp(
        summary="Track how risk and return metrics change over time with rolling windows.",
        steps=[
            "Enter a ticker and set the rolling window size.",
            "Compare rolling metrics against a benchmark.",
        ],
        key_terms=["Rolling Sharpe", "Beta"],
        workflow_position="Standalone",
        related_pages=[
            ("Portfolio Analyzer", "Analyze a full portfolio first"),
        ],
        status="scaffolded",
    ),
    "risk_dashboard": PageHelp(
        summary="Portfolio risk snapshot — VaR, CVaR, drawdown, and concentration.",
        steps=[
            "Set VaR confidence level and calculation method.",
            "View risk metrics for your portfolio.",
        ],
        key_terms=["VaR", "CVaR", "Max Drawdown", "HHI"],
        workflow_position="Portfolio → **Risk**",
        related_pages=[
            ("Portfolio Analyzer", "Build a portfolio to analyze here"),
        ],
        status="scaffolded",
    ),
    "live": PageHelp(
        summary="Real-time price watchlist with auto-refreshing feeds.",
        steps=[
            "Enter symbols to track.",
            "Set refresh interval.",
        ],
        key_terms=["VWAP"],
        workflow_position="Standalone",
        related_pages=[
            ("Market Scanner", "Scan for interesting stocks first"),
        ],
        status="scaffolded",
    ),
    "ai_analysis": PageHelp(
        summary="Ask a local LLM for market insights and analysis.",
        steps=[
            "Pick an **analysis type** (market overview, trade ideas, portfolio review, etc.).",
            "Adjust **temperature** to control creativity (lower = more focused).",
            "Optionally write a **custom prompt** for specific questions.",
            "Click **Generate Analysis** and review the response.",
        ],
        key_terms=["LLM"],
        workflow_position="Standalone",
        related_pages=[
            ("Market Scanner", "Get data to ask questions about"),
            ("Portfolio Analyzer", "Analyze a portfolio before asking for a review"),
        ],
    ),
}


# ── Render Function ──────────────────────────────────────────────────


def render_page_help(page_key: str) -> None:
    """Render a collapsible help expander for the given page.

    Call this in each view's render() function right after the page header.
    """
    import streamlit as st

    help_data = PAGE_HELP.get(page_key)
    if not help_data:
        return

    with st.expander("Page Guide", expanded=False):
        st.markdown(f"**{help_data.summary}**")

        # Workflow position
        if help_data.workflow_position:
            st.markdown(
                f'<div class="page-help-workflow">{help_data.workflow_position}</div>',
                unsafe_allow_html=True,
            )

        # Steps
        st.markdown("**How to use this page:**")
        for i, step in enumerate(help_data.steps, 1):
            st.markdown(f"{i}. {step}")

        # Key terms
        if help_data.key_terms:
            st.markdown("**Key terms on this page:**")
            terms_html = "".join(
                f'<span class="page-help-term" title="{GLOSSARY.get(t, "")}">{t}</span>'
                for t in help_data.key_terms
            )
            st.markdown(
                f'<div class="page-help-terms">{terms_html}</div>',
                unsafe_allow_html=True,
            )

        # Related pages
        if help_data.related_pages:
            st.markdown("**What to do next:**")
            for page_label, reason in help_data.related_pages:
                st.markdown(f"- **{page_label}** — {reason}")
