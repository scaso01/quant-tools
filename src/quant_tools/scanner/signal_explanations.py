"""Structured explanations for scanner signal calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_tools.models import SignalType


@dataclass
class SignalComponent:
    """A single component of a signal calculation."""

    name: str
    weight: float  # fractional weight in overall score (e.g., 0.25)
    description: str
    score_range: tuple[float, float] = (0.0, 100.0)
    interpretation: str = ""


@dataclass
class SignalExplanation:
    """Full explanation for a signal type."""

    name: str
    summary: str
    components: list[SignalComponent] = field(default_factory=list)
    score_interpretation: dict[str, str] = field(default_factory=dict)
    asset_class_notes: dict[str, str] = field(default_factory=dict)


# Build explanations matching the actual signal implementations in signals.py
SIGNAL_EXPLANATIONS: dict[SignalType, SignalExplanation] = {
    SignalType.MOMENTUM: SignalExplanation(
        name="Momentum",
        summary="Measures price trend strength using technical indicators and recent returns.",
        components=[
            SignalComponent(
                name="RSI(14)",
                weight=0.25,
                description="Relative Strength Index over 14 periods. Measures speed and magnitude of recent price changes.",
                interpretation="50-70 is bullish; >70 overbought; <30 oversold",
            ),
            SignalComponent(
                name="SMA Crossover",
                weight=0.25,
                description="20-day vs 50-day Simple Moving Average spread. Positive spread indicates uptrend.",
                interpretation="Positive = bullish trend, negative = bearish trend",
            ),
            SignalComponent(
                name="1-Month Return",
                weight=0.25,
                description="Price return over the last ~21 trading days.",
                interpretation="Higher positive return = stronger momentum",
            ),
            SignalComponent(
                name="Volume Surge",
                weight=0.25,
                description="Current volume relative to 20-day average. High volume confirms price moves.",
                interpretation="Bonus applied when volume exceeds average; absent if no volume data",
            ),
        ],
        score_interpretation={
            "80-100": "Very strong momentum — strong uptrend confirmed by multiple indicators",
            "60-80": "Moderate momentum — price trending higher with some confirmation",
            "40-60": "Neutral — no clear directional bias",
            "20-40": "Weak/negative momentum — price trending lower",
            "0-20": "Strong downtrend — all indicators bearish",
        },
        asset_class_notes={
            "equity": "Standard RSI bands and SMA periods",
            "crypto": "Wider RSI bands (crypto is more volatile), stronger volume emphasis (24/7 markets)",
            "commodity": "Mean-reversion overlay: distance from 50-day SMA + Bollinger position",
        },
    ),
    SignalType.VALUE: SignalExplanation(
        name="Value",
        summary="Evaluates fundamental cheapness using valuation ratios.",
        components=[
            SignalComponent(
                name="P/E Ratio",
                weight=0.4,
                description="Price-to-Earnings ratio. Lower values indicate cheaper stocks.",
                interpretation="<10 = deep value (score 100), >40 = expensive (score 0)",
            ),
            SignalComponent(
                name="P/B Ratio",
                weight=0.4,
                description="Price-to-Book ratio. Compares market price to book value.",
                interpretation="<1 = trading below book (score 100), >10 = very expensive (score 0)",
            ),
            SignalComponent(
                name="Forward PE Bonus",
                weight=0.2,
                description="Bonus if forward P/E is lower than trailing P/E, indicating expected earnings growth.",
                interpretation="Forward < Trailing = positive catalyst for re-rating",
            ),
        ],
        score_interpretation={
            "80-100": "Deep value — significantly undervalued on multiple metrics",
            "60-80": "Moderate value — reasonably priced with some margin of safety",
            "40-60": "Fair value — neither cheap nor expensive",
            "20-40": "Expensive — trading at premium valuations",
            "0-20": "Very expensive — extreme premium, high expectations priced in",
        },
        asset_class_notes={
            "equity": "Uses P/E, P/B, forward P/E from fundamental data",
            "crypto": "Not applicable — no fundamental ratios available",
            "commodity": "Not applicable — commodities don't have earnings",
        },
    ),
    SignalType.QUALITY: SignalExplanation(
        name="Quality",
        summary="Assesses business quality through profitability and financial health metrics.",
        components=[
            SignalComponent(
                name="Return on Equity",
                weight=0.33,
                description="ROE measures how efficiently a company uses shareholder equity to generate profit.",
                interpretation=">25% = excellent (score 100), <0% = value-destroying (score 0)",
            ),
            SignalComponent(
                name="Profit Margin",
                weight=0.33,
                description="Net profit margin — the percentage of revenue retained as profit.",
                interpretation="30%+ = highly profitable (score 100), 0% = breakeven (score 0)",
            ),
            SignalComponent(
                name="Debt/Equity",
                weight=0.34,
                description="Financial leverage ratio. Lower is healthier.",
                interpretation="0 = no debt (score 100), >200% = heavily leveraged (score 0)",
            ),
        ],
        score_interpretation={
            "80-100": "Exceptional quality — highly profitable, low debt, strong ROE",
            "60-80": "Good quality — solid profitability and manageable leverage",
            "40-60": "Average quality — mixed financial health indicators",
            "20-40": "Below average — weak profitability or high leverage",
            "0-20": "Poor quality — unprofitable or dangerously leveraged",
        },
        asset_class_notes={
            "equity": "Full fundamental analysis using ROE, margins, D/E",
            "crypto": "Not applicable — no company fundamentals",
            "commodity": "Not applicable — physical goods have no balance sheet",
        },
    ),
    SignalType.MEAN_REVERSION: SignalExplanation(
        name="Mean Reversion",
        summary="Identifies oversold conditions where price may revert to mean.",
        components=[
            SignalComponent(
                name="Bollinger %B",
                weight=0.5,
                description="Position within Bollinger Bands (20-day, 2 std). Near lower band suggests oversold.",
                interpretation="Near 0 = at lower band (oversold, high score); near 1 = at upper band (overbought, low score)",
            ),
            SignalComponent(
                name="RSI Extremes",
                weight=0.5,
                description="RSI at extreme levels (<30 or >70) signals potential mean reversion opportunity.",
                interpretation="<30 = oversold (high score); >70 = overbought (moderate score for short bias)",
            ),
        ],
        score_interpretation={
            "80-100": "Strongly oversold — high probability of upward reversion",
            "60-80": "Moderately oversold — price stretched below mean",
            "40-60": "Near fair value — limited reversion opportunity",
            "20-40": "Moderately overbought — possible downward reversion",
            "0-20": "Strongly overbought — price well above mean",
        },
        asset_class_notes={
            "equity": "Standard Bollinger (20d, 2 std) and RSI(14)",
            "crypto": "Wider bands due to higher volatility",
            "commodity": "Commodities exhibit strong mean-reversion tendencies",
        },
    ),
}


def get_signal_explanation(signal_type: SignalType) -> SignalExplanation:
    """Get the explanation for a signal type."""
    return SIGNAL_EXPLANATIONS[signal_type]
