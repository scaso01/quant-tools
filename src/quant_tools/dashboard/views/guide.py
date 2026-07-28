"""Quant-Tools Guide — central help hub."""

from __future__ import annotations

import streamlit as st

from quant_tools.dashboard.core.page_help import GLOSSARY


def render() -> None:
    """Render the full guide page with workflow map, FAQ, and glossary."""

    # Close button
    if st.button("← Back to Dashboard", key="close_guide"):
        st.session_state["show_guide"] = False
        st.rerun()

    st.markdown(
        '<div class="dashboard-header">'
        '<div class="dashboard-title">Quant-Tools Guide</div>'
        '<div class="dashboard-subtitle">Everything you need to know about using this dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Section 1: Overview ──────────────────────────────────────────
    st.markdown("### What is Quant-Tools?")
    st.markdown(
        "Quant-Tools is a **quantitative finance toolkit** for retail investors and market enthusiasts. "
        "It helps you scan markets for opportunities, test trading strategies on historical data, "
        "build and analyze portfolios, and track ML-powered predictions — all without writing code."
    )
    st.markdown(
        "The dashboard has **10 pages** organized into five groups: "
        "**Discover** (find opportunities), **Build** (test and construct), "
        "**Analyze** (score and predict), **Monitor** (track risk and prices), "
        "and **Tools** (AI-powered insights)."
    )

    st.markdown("---")

    # ── Section 2: Workflow Map ──────────────────────────────────────
    st.markdown("### How the Pages Connect")
    st.markdown(
        "Pages are designed to feed into each other. Here's how a typical workflow flows:"
    )

    # HTML/CSS workflow diagram
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:16px;padding:16px 0;">
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <div style="background:var(--bg-card);border:1px solid var(--accent-green);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-green);">
                    📡 Scanner
                </div>
                <span style="color:var(--text-muted);font-size:1.2rem;">→</span>
                <div style="background:var(--bg-card);border:1px solid var(--accent-blue);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-blue);">
                    📈 Backtester
                </div>
                <span style="color:var(--text-muted);font-size:1.2rem;">→</span>
                <div style="background:var(--bg-card);border:1px solid var(--accent-blue);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-blue);">
                    💼 Portfolio
                </div>
                <span style="color:var(--text-muted);font-size:1.2rem;">→</span>
                <div style="background:var(--bg-card);border:1px solid var(--accent-amber);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-amber);">
                    🛡️ Risk
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <div style="background:var(--bg-card);border:1px solid var(--accent-purple);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-purple);">
                    🏛️ Factors
                </div>
                <span style="color:var(--text-muted);font-size:1.2rem;">→</span>
                <div style="background:var(--bg-card);border:1px solid var(--accent-purple);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-purple);">
                    🔮 Predictions
                </div>
                <span style="color:var(--text-muted);font-size:1.2rem;">→</span>
                <div style="background:var(--bg-card);border:1px solid var(--accent-purple);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--accent-purple);">
                    📊 Performance
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <div style="background:var(--bg-card);border:1px solid var(--text-dim);border-radius:8px;padding:8px 16px;font-weight:600;color:var(--text-muted);">
                    🤖 AI Analysis — standalone, works with any page
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 3: Suggested Workflows ───────────────────────────────
    st.markdown("### Suggested Workflows")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-value" style="color:var(--accent-green);font-size:1.1rem;">New to Investing?</div>'
            '<div class="metric-card-label" style="text-transform:none;font-size:0.8rem;line-height:1.5;">'
            '1. Scan for stocks with strong signals<br>'
            '2. Backtest a strategy on your picks<br>'
            '3. Build a portfolio and check risk'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-value" style="color:var(--accent-blue);font-size:1.1rem;">Research a Stock</div>'
            '<div class="metric-card-label" style="text-transform:none;font-size:0.8rem;line-height:1.5;">'
            '1. Scan to find the stock\'s signals<br>'
            '2. Check ML predictions for it<br>'
            '3. Ask AI Analysis for insights'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-value" style="color:var(--accent-purple);font-size:1.1rem;">Macro Analysis</div>'
            '<div class="metric-card-label" style="text-transform:none;font-size:0.8rem;line-height:1.5;">'
            '1. Check the macro regime in Factors<br>'
            '2. Score stocks by factor exposure<br>'
            '3. Track predictions over time'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section 4: FAQ ───────────────────────────────────────────────
    st.markdown("### Frequently Asked Questions")

    with st.expander("Why did a stock score high but its price dropped?"):
        st.markdown(
            "Signal scores measure **technical patterns**, not future price guarantees. "
            "A high momentum score means the stock has been trending up recently, but markets "
            "can reverse at any time. Signals are tools for identifying candidates, not crystal balls. "
            "Always combine signals with your own research and risk management."
        )

    with st.expander("Which backtesting strategy should I start with?"):
        st.markdown(
            "Start with **SMA Crossover** — it's the simplest and most intuitive. "
            "It buys when a short-term moving average crosses above a long-term one (bullish trend) "
            "and sells when it crosses below (bearish trend). Once you understand the results, "
            "try Bollinger Bands or RSI for more nuanced strategies."
        )

    with st.expander("What does the Score number mean? Is 75 good?"):
        st.markdown(
            "Scores are **relative rankings** from 0 to 100 within the scanned universe. "
            "A score of 75 means the stock ranks in the top 25% for that signal type. "
            "Scores are NOT predictions — they summarize how strongly a stock exhibits "
            "the selected trading signal compared to its peers."
        )

    with st.expander("How accurate are the ML predictions?"):
        st.markdown(
            "Check the **Performance Tracker** page for actual win rates. "
            "The ensemble model (which combines 12 individual models) typically achieves "
            "55-65% accuracy depending on the asset class and time horizon. "
            "Predictions are probabilistic — they show the direction and confidence, "
            "not a guaranteed outcome."
        )

    with st.expander("Where does the data come from?"):
        st.markdown(
            "- **Stock prices**: Yahoo Finance (via yfinance) and Stooq\n"
            "- **Crypto prices**: CCXT (connecting to multiple exchanges)\n"
            "- **Macro data**: Federal Reserve Economic Data (FRED)\n"
            "- **News**: Yahoo Finance RSS feeds\n"
            "- **AI Analysis**: Your local LLM (llama-server on port 8080)\n\n"
            "All data sources are free and don't require API keys (except FRED for macro regime detection)."
        )

    st.markdown("---")

    # ── Section 5: Searchable Glossary ───────────────────────────────
    st.markdown("### Glossary")

    search = st.text_input(
        "Search terms...",
        key="glossary_search",
        placeholder="Type to filter (e.g., Sharpe, VaR, Monte Carlo)",
    )

    search_lower = search.lower().strip()
    filtered = {
        k: v for k, v in sorted(GLOSSARY.items())
        if not search_lower or search_lower in k.lower() or search_lower in v.lower()
    }

    if not filtered:
        st.info("No matching terms found.")
    else:
        for term, definition in filtered.items():
            st.markdown(f"**{term}** — {definition}")

    # ── Section 6: Page Status ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### Page Status")
    st.markdown(
        "| Page | Status |\n"
        "|------|--------|\n"
        "| 📡 Market Scanner | Active |\n"
        "| 📈 Strategy Backtester | Active |\n"
        "| 💼 Portfolio Analyzer | Active |\n"
        "| 🏛️ Factors & Macro | Active |\n"
        "| 🔮 Predictions | Active |\n"
        "| 📊 Performance | Active |\n"
        "| 📉 Rolling Analytics | In Development |\n"
        "| 🛡️ Risk Dashboard | In Development |\n"
        "| ⚡ Live Monitor | In Development |\n"
        "| 🤖 AI Analysis | Active (requires local LLM) |"
    )
