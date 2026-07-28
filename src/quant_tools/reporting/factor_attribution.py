"""Factor attribution — decompose portfolio returns into factor exposures.

Uses OLS regression against factor returns (e.g., Fama-French 3-factor).
Falls back to CAPM (market-only) when additional factors are unavailable.
All computations are pure numpy/pandas — no network calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class FactorExposure:
    """Single factor exposure result."""

    name: str
    coefficient: float
    t_stat: float = 0.0
    p_value: float = 1.0


@dataclass
class AttributionResult:
    """Full factor attribution result."""

    alpha: float = 0.0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    exposures: list[FactorExposure] = field(default_factory=list)
    residual_returns: Optional[pd.Series] = None
    model_type: str = "ols"  # "ols" | "capm"


# =============================================================================
# OLS regression helper
# =============================================================================


def _ols_regression(
    y: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Ordinary Least Squares via the normal equation.

    Parameters
    ----------
    y : (n,) dependent variable
    X : (n, k) independent variables (WITHOUT intercept — added internally)

    Returns
    -------
    coefficients : (k+1,) — intercept first, then factor betas
    r_squared : float
    adj_r_squared : float
    t_stats : (k+1,)
    p_values : (k+1,)
    """
    n = len(y)
    # Add intercept column
    ones = np.ones((n, 1))
    X_full = np.hstack([ones, X]) if X.ndim == 2 else np.hstack([ones, X.reshape(-1, 1)])
    k = X_full.shape[1]

    # Normal equation: beta = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
        # Singular matrix — return zeros
        zeros = np.zeros(k)
        return zeros, 0.0, 0.0, zeros, np.ones(k)

    coeffs = XtX_inv @ X_full.T @ y

    # Residuals and R-squared
    y_hat = X_full @ coeffs
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / (n - k)
        if n > k
        else 0.0
    )

    # Standard errors and t-stats
    if n > k and ss_res > 0:
        mse = ss_res / (n - k)
        se = np.sqrt(np.diag(XtX_inv) * mse)
        se = np.where(se > 0, se, 1e-10)
        t_stats = coeffs / se
    else:
        t_stats = np.zeros(k)

    # Approximate p-values via normal distribution (avoids scipy dependency)
    # For |t| > 6, p ≈ 0; for moderate t, use rough approximation
    p_values = np.exp(-0.5 * t_stats ** 2) * 2  # Very rough two-tailed
    p_values = np.clip(p_values, 0.0, 1.0)

    return coeffs, r_squared, adj_r_squared, t_stats, p_values


# =============================================================================
# Main decomposition
# =============================================================================


def decompose(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> AttributionResult:
    """Decompose portfolio returns into factor exposures via OLS.

    Parameters
    ----------
    portfolio_returns:
        Daily simple returns.
    factor_returns:
        DataFrame with columns as factor names, same date index as returns.
    risk_free_rate:
        Annual risk-free rate.
    periods_per_year:
        Trading days per year.

    Returns
    -------
    AttributionResult with factor exposures, R-squared, and alpha.
    """
    if portfolio_returns.empty or factor_returns.empty:
        logger.warning("Empty input — returning blank attribution")
        return AttributionResult()

    # Align dates
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), factor_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 3:
        logger.warning("Insufficient overlapping data — returning blank attribution")
        return AttributionResult()

    y = aligned["portfolio"].values
    factor_cols = [c for c in aligned.columns if c != "portfolio"]
    X = aligned[factor_cols].values

    coeffs, r_sq, adj_r_sq, t_stats, p_vals = _ols_regression(y, X)

    # Build exposures (coeffs[0] is intercept = alpha)
    exposures = []
    for i, name in enumerate(factor_cols):
        exposures.append(
            FactorExposure(
                name=name,
                coefficient=float(coeffs[i + 1]),
                t_stat=float(t_stats[i + 1]),
                p_value=float(p_vals[i + 1]),
            )
        )

    # Annualize alpha
    daily_alpha = float(coeffs[0])
    annual_alpha = daily_alpha * periods_per_year

    # Residuals
    y_hat = np.hstack([np.ones((len(y), 1)), X]) @ coeffs
    residuals = pd.Series(y - y_hat, index=aligned.index, name="residual")

    return AttributionResult(
        alpha=annual_alpha,
        r_squared=float(r_sq),
        adj_r_squared=float(adj_r_sq),
        exposures=exposures,
        residual_returns=residuals,
        model_type="ols",
    )


def capm_decompose(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> AttributionResult:
    """CAPM single-factor decomposition (fallback when factors unavailable).

    Parameters
    ----------
    portfolio_returns:
        Daily simple returns.
    market_returns:
        Daily market returns (e.g., SPY).
    risk_free_rate:
        Annual risk-free rate.
    periods_per_year:
        Trading days per year.

    Returns
    -------
    AttributionResult with single market exposure.
    """
    factor_df = market_returns.to_frame("Market")
    result = decompose(
        portfolio_returns, factor_df, risk_free_rate, periods_per_year
    )
    result.model_type = "capm"
    return result
