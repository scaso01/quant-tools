"""Enhanced risk analysis: VaR trio, stress testing, tail risk, correlation breakdown."""

from __future__ import annotations

import logging

import numpy as np
from dataclasses import dataclass
from scipy import stats as scipy_stats

log = logging.getLogger("quant_tools")


# ── Data Models ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class VaRResult:
    """Value at Risk computed via three methods."""

    parametric: float
    historical: float
    monte_carlo: float
    confidence: float
    backtest_pvalue: float | None = None


@dataclass(frozen=True)
class StressScenario:
    """Definition of a market stress scenario."""

    name: str
    equity_shock: float  # e.g., -0.30 for -30%
    rate_shock_bps: float  # e.g., 200 for +200bps
    vol_shock: float  # e.g., 0.80 for VIX to 80
    description: str = ""


@dataclass(frozen=True)
class TailRiskMetrics:
    """Comprehensive tail risk measurements."""

    evar: float  # Entropic VaR
    cdar: float  # Conditional Drawdown at Risk
    max_dd_duration: int  # Longest peak-to-recovery in periods
    tail_ratio: float  # 95th percentile gain / |5th percentile loss|


# ── Historical Stress Scenarios ──────────────────────────────────────


HISTORICAL_SCENARIOS: dict[str, StressScenario] = {
    "2008_gfc": StressScenario(
        "2008 GFC", -0.50, 0, 0.80, "Global Financial Crisis"
    ),
    "2020_covid": StressScenario(
        "2020 COVID", -0.34, -150, 0.82, "COVID crash"
    ),
    "2022_rates": StressScenario(
        "2022 Rate Shock", -0.25, 400, 0.35, "Fed rate hiking cycle"
    ),
    "2023_banking": StressScenario(
        "2023 Banking", -0.10, 0, 0.30, "SVB/regional bank crisis"
    ),
}


# ── Enhanced Risk Analyzer ───────────────────────────────────────────


class EnhancedRiskAnalyzer:
    """Comprehensive risk analysis: VaR trio, stress testing, tail risk."""

    def var_trio(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        n_sims: int = 10_000,
        seed: int | None = 42,
    ) -> VaRResult:
        """Compute VaR via parametric, historical, and Monte Carlo methods.

        Args:
            returns: 1-D array of portfolio returns.
            confidence: Confidence level (e.g. 0.95 for 95% VaR).
            n_sims: Number of Monte Carlo simulations.
            seed: Random seed for MC reproducibility.

        Returns:
            VaRResult with all three VaR estimates (as negative losses).
        """
        returns = np.asarray(returns, dtype=np.float64)
        alpha = 1.0 - confidence

        # Parametric (normal assumption): VaR = mu + sigma * z_alpha
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        z = scipy_stats.norm.ppf(alpha)
        parametric = float(mu + sigma * z)

        # Historical: empirical percentile
        historical = float(np.percentile(returns, alpha * 100))

        # Monte Carlo: simulate from fitted normal, take percentile
        rng = np.random.default_rng(seed)
        simulated = rng.normal(mu, sigma, n_sims)
        monte_carlo = float(np.percentile(simulated, alpha * 100))

        return VaRResult(
            parametric=round(parametric, 8),
            historical=round(historical, 8),
            monte_carlo=round(monte_carlo, 8),
            confidence=confidence,
        )

    def backtest_var(
        self,
        returns: np.ndarray,
        var_estimates: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Kupiec POF test + Christoffersen independence test.

        Tests whether the number and clustering of VaR violations
        are consistent with the stated confidence level.

        Args:
            returns: Realized returns (1-D array).
            var_estimates: Corresponding VaR estimates (same length).
            confidence: Stated confidence level.

        Returns:
            Dict with kupiec_pvalue, christoffersen_pvalue,
            n_violations, expected_violations.
        """
        returns = np.asarray(returns, dtype=np.float64)
        var_estimates = np.asarray(var_estimates, dtype=np.float64)
        n = len(returns)
        alpha = 1.0 - confidence

        # Violations: return < VaR
        violations = (returns < var_estimates).astype(int)
        n_violations = int(np.sum(violations))
        expected = alpha * n

        # Kupiec Proportion of Failures (POF) test
        # LR_pof = -2 * [T1*log(alpha) + T0*log(1-alpha)
        #               - T1*log(T1/T) - T0*log(T0/T)]
        t1 = n_violations
        t0 = n - t1

        if t1 == 0 or t1 == n:
            kupiec_pvalue = 0.0 if abs(t1 / n - alpha) > 0.1 else 1.0
        else:
            pi_hat = t1 / n
            lr_pof = -2.0 * (
                t1 * np.log(alpha) + t0 * np.log(1.0 - alpha)
                - t1 * np.log(pi_hat) - t0 * np.log(1.0 - pi_hat)
            )
            kupiec_pvalue = float(1.0 - scipy_stats.chi2.cdf(lr_pof, df=1))

        # Christoffersen independence test
        # Count transition pairs: 00, 01, 10, 11
        n00, n01, n10, n11 = 0, 0, 0, 0
        for i in range(1, len(violations)):
            prev, curr = violations[i - 1], violations[i]
            if prev == 0 and curr == 0:
                n00 += 1
            elif prev == 0 and curr == 1:
                n01 += 1
            elif prev == 1 and curr == 0:
                n10 += 1
            else:
                n11 += 1

        # Under independence, transition probs should be equal
        row0 = n00 + n01
        row1 = n10 + n11

        if row0 > 0 and row1 > 0 and n01 > 0 and n10 > 0:
            pi01 = n01 / row0
            pi11 = n11 / row1 if row1 > 0 else 0.0
            pi_hat_ind = (n01 + n11) / (n00 + n01 + n10 + n11)

            # Log-likelihood under independence
            ll_ind = (
                n00 * np.log(1.0 - pi_hat_ind + 1e-15)
                + n01 * np.log(pi_hat_ind + 1e-15)
                + n10 * np.log(1.0 - pi_hat_ind + 1e-15)
                + n11 * np.log(pi_hat_ind + 1e-15)
            )
            # Log-likelihood under dependence
            ll_dep = (
                n00 * np.log(1.0 - pi01 + 1e-15)
                + n01 * np.log(pi01 + 1e-15)
                + n10 * np.log(1.0 - pi11 + 1e-15)
                + n11 * np.log(pi11 + 1e-15)
            )
            lr_ind = -2.0 * (ll_ind - ll_dep)
            christoffersen_pvalue = float(
                1.0 - scipy_stats.chi2.cdf(max(lr_ind, 0.0), df=1)
            )
        else:
            christoffersen_pvalue = 1.0

        return {
            "kupiec_pvalue": round(kupiec_pvalue, 6),
            "christoffersen_pvalue": round(christoffersen_pvalue, 6),
            "n_violations": n_violations,
            "expected_violations": round(expected, 2),
        }

    def stress_test(
        self,
        portfolio_returns: np.ndarray,
        portfolio_weights: dict[str, float],
        scenario: StressScenario | str,
    ) -> dict:
        """Apply stress scenario to portfolio.

        For named scenarios (string), looks up HISTORICAL_SCENARIOS.
        Applies the equity shock as a multiplicative factor to portfolio returns.

        Args:
            portfolio_returns: Historical returns array.
            portfolio_weights: {symbol: weight} dict.
            scenario: StressScenario instance or key into HISTORICAL_SCENARIOS.

        Returns:
            Dict with scenario name, portfolio_loss, description.
        """
        if isinstance(scenario, str):
            if scenario not in HISTORICAL_SCENARIOS:
                raise ValueError(
                    f"Unknown scenario: {scenario}. "
                    f"Choose from {list(HISTORICAL_SCENARIOS)}"
                )
            scenario = HISTORICAL_SCENARIOS[scenario]

        portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)

        # Total equity weight
        total_weight = sum(portfolio_weights.values())
        equity_exposure = total_weight  # simplified: assume all equity

        # Apply equity shock proportionally
        portfolio_loss = float(equity_exposure * scenario.equity_shock)

        # Volatility amplification: scale historical vol by shock
        hist_vol = float(np.std(portfolio_returns, ddof=1))
        stressed_vol = hist_vol * max(scenario.vol_shock / 0.20, 1.0)

        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "portfolio_loss": round(portfolio_loss, 6),
            "equity_shock": scenario.equity_shock,
            "rate_shock_bps": scenario.rate_shock_bps,
            "vol_shock": scenario.vol_shock,
            "stressed_vol": round(stressed_vol, 6),
            "hist_vol": round(hist_vol, 6),
        }

    def correlation_breakdown(self, returns_matrix: np.ndarray) -> dict:
        """Eigenvalue analysis for correlation convergence detection.

        During crises, all asset correlations converge toward 1.
        This is detectable by examining the first eigenvalue of the
        correlation matrix -- if it accounts for >60% of total variance,
        assets are moving together (crisis regime).

        Args:
            returns_matrix: 2-D array of shape (n_obs, n_assets).

        Returns:
            Dict with first_eigenvalue_ratio, is_converging, eigenvalues.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)

        if returns_matrix.ndim != 2 or returns_matrix.shape[1] < 2:
            return {
                "first_eigenvalue_ratio": 0.0,
                "is_converging": False,
                "eigenvalues": [],
            }

        # Compute correlation matrix
        # Handle constant columns (zero std) by adding tiny noise
        stds = np.std(returns_matrix, axis=0, ddof=1)
        if np.any(stds < 1e-15):
            returns_matrix = returns_matrix.copy()
            returns_matrix[:, stds < 1e-15] += np.random.default_rng(0).normal(
                0, 1e-10, (returns_matrix.shape[0], int(np.sum(stds < 1e-15)))
            )

        corr = np.corrcoef(returns_matrix, rowvar=False)

        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending

        total_variance = float(np.sum(eigenvalues))
        if total_variance <= 0:
            return {
                "first_eigenvalue_ratio": 0.0,
                "is_converging": False,
                "eigenvalues": eigenvalues.tolist(),
            }

        first_ratio = float(eigenvalues[0] / total_variance)

        return {
            "first_eigenvalue_ratio": round(first_ratio, 6),
            "is_converging": first_ratio > 0.60,
            "eigenvalues": [round(float(v), 6) for v in eigenvalues],
        }

    def tail_metrics(self, returns: np.ndarray) -> TailRiskMetrics:
        """Comprehensive tail risk metrics.

        Args:
            returns: 1-D array of portfolio returns.

        Returns:
            TailRiskMetrics with EVaR, CDaR, max DD duration, tail ratio.
        """
        returns = np.asarray(returns, dtype=np.float64)

        # ── EVaR (Entropic Value at Risk) ──
        # Exponential Chebyshev bound: EVaR_alpha = inf_t>0 {log(E[exp(-t*X)]) / t + log(alpha) / t}
        # Simplified: use moment-generating approximation
        # EVaR >= VaR and is an upper bound on CVaR
        alpha = 0.05
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma > 0:
            # Analytical EVaR for normal distribution:
            # EVaR = mu - sigma * sqrt(-2 * log(alpha))
            evar = float(mu - sigma * np.sqrt(-2.0 * np.log(alpha)))
        else:
            evar = mu

        # ── CDaR (Conditional Drawdown at Risk) ──
        # First compute drawdown series, then take conditional mean beyond DD-VaR
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max  # All <= 0

        if len(drawdowns) > 0:
            dd_var = float(np.percentile(drawdowns, alpha * 100))
            tail_dd = drawdowns[drawdowns <= dd_var]
            cdar = float(np.mean(tail_dd)) if len(tail_dd) > 0 else dd_var
        else:
            cdar = 0.0

        # ── Max Drawdown Duration ──
        max_dd_duration = self._max_dd_duration(cumulative, running_max)

        # ── Tail Ratio ──
        # 95th percentile gain / |5th percentile loss|
        p95 = float(np.percentile(returns, 95))
        p5 = float(np.percentile(returns, 5))
        tail_ratio = abs(p95 / p5) if abs(p5) > 1e-15 else 0.0

        return TailRiskMetrics(
            evar=round(evar, 8),
            cdar=round(cdar, 8),
            max_dd_duration=max_dd_duration,
            tail_ratio=round(tail_ratio, 6),
        )

    @staticmethod
    def _max_dd_duration(
        cumulative: np.ndarray,
        running_max: np.ndarray,
    ) -> int:
        """Find longest peak-to-recovery period in the cumulative series."""
        in_drawdown = cumulative < running_max
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration
