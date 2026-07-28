"""Portfolio analysis module."""

from __future__ import annotations

from quant_tools.portfolio.analyzer import (
    analyze_portfolio,
    analyze_portfolio_with_benchmark,
    optimize_portfolio,
)
from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS, BenchmarkAnalyzer
from quant_tools.portfolio.black_litterman import BlackLitterman
from quant_tools.portfolio.cross_correlation import (
    calculate_cross_asset_correlations,
    calculate_rolling_correlations,
)
from quant_tools.portfolio.hrp import HERC, NCO, HierarchicalRiskParity, plot_dendrogram
from quant_tools.portfolio.monte_carlo import (
    estimate_distribution_params,
    fit_best_distribution,
    run_monte_carlo,
    run_monte_carlo_garch,
)
from quant_tools.portfolio.optimizer import PortfolioOptimizer
from quant_tools.portfolio.regime import RegimeDetector, RegimeState
from quant_tools.portfolio.risk import RiskAnalyzer
from quant_tools.portfolio.risk_enhanced import (
    HISTORICAL_SCENARIOS,
    EnhancedRiskAnalyzer,
    StressScenario,
    TailRiskMetrics,
    VaRResult,
)

__all__ = [
    "analyze_portfolio",
    "analyze_portfolio_with_benchmark",
    "calculate_cross_asset_correlations",
    "calculate_rolling_correlations",
    "optimize_portfolio",
    "estimate_distribution_params",
    "fit_best_distribution",
    "run_monte_carlo",
    "run_monte_carlo_garch",
    "BenchmarkAnalyzer",
    "BlackLitterman",
    "EnhancedRiskAnalyzer",
    "HERC",
    "HISTORICAL_SCENARIOS",
    "HierarchicalRiskParity",
    "NCO",
    "PortfolioOptimizer",
    "RegimeDetector",
    "RegimeState",
    "RiskAnalyzer",
    "STANDARD_BENCHMARKS",
    "StressScenario",
    "TailRiskMetrics",
    "VaRResult",
    "plot_dendrogram",
]
