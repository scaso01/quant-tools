"""Tearsheet generation — compute portfolio metrics and produce HTML reports.

Provides TearsheetData (metrics container) and TearsheetGenerator (HTML builder).
All calculations are pure numpy/pandas — no network calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data container
# =============================================================================


@dataclass
class TearsheetData:
    """Container for computed tearsheet metrics."""

    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    monthly_returns: Optional[pd.DataFrame] = None
    yearly_returns: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    benchmark_metrics: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Metrics computation
# =============================================================================


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> TearsheetData:
    """Compute tearsheet metrics from a return series.

    Parameters
    ----------
    returns:
        Daily simple returns (not cumulative).
    benchmark_returns:
        Optional benchmark return series for relative metrics.
    risk_free_rate:
        Annual risk-free rate (default 0).
    periods_per_year:
        Trading days per year (default 252).

    Returns
    -------
    TearsheetData with all fields populated.
    """
    if returns.empty:
        logger.warning("Empty returns series — returning zero metrics")
        return TearsheetData()

    data = TearsheetData()

    # ── Core metrics ──
    cumulative = (1 + returns).cumprod()
    data.total_return = float(cumulative.iloc[-1] - 1.0)

    n_years = len(returns) / periods_per_year
    if n_years > 0 and cumulative.iloc[-1] > 0:
        data.cagr = float(cumulative.iloc[-1] ** (1.0 / n_years) - 1.0)
    else:
        data.cagr = 0.0

    data.volatility = float(returns.std() * np.sqrt(periods_per_year))

    # Sharpe
    excess = returns - risk_free_rate / periods_per_year
    if returns.std() > 0:
        data.sharpe_ratio = float(
            excess.mean() / returns.std() * np.sqrt(periods_per_year)
        )
    else:
        data.sharpe_ratio = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        data.sortino_ratio = float(
            excess.mean() / downside.std() * np.sqrt(periods_per_year)
        )
    else:
        data.sortino_ratio = 0.0

    # Drawdown
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    data.max_drawdown = float(abs(drawdown.min()))
    data.drawdown_series = drawdown

    # Calmar
    if data.max_drawdown > 0:
        data.calmar_ratio = float(data.cagr / data.max_drawdown)
    else:
        data.calmar_ratio = 0.0

    # ── Monthly / yearly aggregation ──
    if hasattr(returns.index, "to_period"):
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if not monthly.empty:
            # Pivot: rows=year, cols=month
            df = monthly.to_frame("return")
            df["year"] = df.index.year
            df["month"] = df.index.month
            data.monthly_returns = df.pivot(
                index="year", columns="month", values="return"
            )

        yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        data.yearly_returns = yearly

    # ── Benchmark-relative metrics ──
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat(
            [returns, benchmark_returns], axis=1, join="inner"
        )
        if len(aligned) > 1:
            r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
            cov_matrix = np.cov(r.values, b.values)
            if cov_matrix[1, 1] > 0:
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
            else:
                beta = 0.0
            alpha = float(
                (r.mean() - risk_free_rate / periods_per_year)
                - beta * (b.mean() - risk_free_rate / periods_per_year)
            ) * periods_per_year
            tracking_error = float((r - b).std() * np.sqrt(periods_per_year))
            info_ratio = (
                float((r - b).mean() / (r - b).std() * np.sqrt(periods_per_year))
                if (r - b).std() > 0
                else 0.0
            )
            data.benchmark_metrics = {
                "alpha": alpha,
                "beta": beta,
                "tracking_error": tracking_error,
                "information_ratio": info_ratio,
            }

    return data


# =============================================================================
# HTML generator
# =============================================================================


class TearsheetGenerator:
    """Generate an HTML tearsheet from TearsheetData."""

    SECTIONS = [
        "summary",
        "returns",
        "drawdown",
        "monthly_table",
        "benchmark",
    ]

    def __init__(self, title: str = "Portfolio Tearsheet") -> None:
        self.title = title

    def generate_html(self, data: TearsheetData) -> str:
        """Produce a complete HTML tearsheet string."""
        sections: list[str] = []
        sections.append(self._header())
        sections.append(self._summary_section(data))
        sections.append(self._returns_section(data))
        sections.append(self._drawdown_section(data))
        if data.monthly_returns is not None:
            sections.append(self._monthly_table_section(data))
        if data.benchmark_metrics:
            sections.append(self._benchmark_section(data))
        sections.append(self._footer())
        return "\n".join(sections)

    # ── Section builders ──

    def _header(self) -> str:
        return (
            "<!DOCTYPE html><html><head>"
            f"<title>{self.title}</title>"
            "<style>body{font-family:sans-serif;margin:20px}"
            "table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:6px;text-align:right}"
            ".section{margin:20px 0}</style>"
            f"</head><body><h1>{self.title}</h1>"
        )

    def _footer(self) -> str:
        return "</body></html>"

    def _summary_section(self, data: TearsheetData) -> str:
        return (
            '<div class="section"><h2>Summary</h2>'
            f"<p>Total Return: {data.total_return:.2%}</p>"
            f"<p>CAGR: {data.cagr:.2%}</p>"
            f"<p>Sharpe Ratio: {data.sharpe_ratio:.2f}</p>"
            f"<p>Sortino Ratio: {data.sortino_ratio:.2f}</p>"
            f"<p>Max Drawdown: {data.max_drawdown:.2%}</p>"
            f"<p>Volatility: {data.volatility:.2%}</p>"
            f"<p>Calmar Ratio: {data.calmar_ratio:.2f}</p>"
            "</div>"
        )

    def _returns_section(self, data: TearsheetData) -> str:
        lines = ['<div class="section"><h2>Returns</h2>']
        if data.yearly_returns is not None and not data.yearly_returns.empty:
            lines.append("<table><tr><th>Year</th><th>Return</th></tr>")
            for dt, val in data.yearly_returns.items():
                yr = dt.year if hasattr(dt, "year") else dt
                lines.append(f"<tr><td>{yr}</td><td>{val:.2%}</td></tr>")
            lines.append("</table>")
        else:
            lines.append("<p>No yearly returns available.</p>")
        lines.append("</div>")
        return "\n".join(lines)

    def _drawdown_section(self, data: TearsheetData) -> str:
        return (
            '<div class="section"><h2>Drawdown</h2>'
            f"<p>Maximum Drawdown: {data.max_drawdown:.2%}</p>"
            "</div>"
        )

    def _monthly_table_section(self, data: TearsheetData) -> str:
        assert data.monthly_returns is not None
        html = data.monthly_returns.to_html(
            float_format=lambda x: f"{x:.2%}", na_rep=""
        )
        return f'<div class="section"><h2>Monthly Returns</h2>{html}</div>'

    def _benchmark_section(self, data: TearsheetData) -> str:
        lines = ['<div class="section"><h2>Benchmark Comparison</h2>']
        for key, val in data.benchmark_metrics.items():
            lines.append(f"<p>{key.replace('_', ' ').title()}: {val:.4f}</p>")
        lines.append("</div>")
        return "\n".join(lines)
