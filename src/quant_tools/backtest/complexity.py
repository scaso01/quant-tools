"""Model complexity assessment tools.

Provides:
- count_free_params: Count tunable parameters in a strategy configuration.
- complexity_budget: Assess data sufficiency for the number of parameters.
- bayesian_information_criterion: BIC for model selection.
"""

from __future__ import annotations

import numpy as np


# Known parameter counts per strategy (default params excluded from count).
# Each entry maps strategy name -> list of parameter names that are tunable.
_STRATEGY_PARAMS: dict[str, list[str]] = {
    "sma_crossover": ["fast", "slow"],
    "bollinger_strategy": ["period", "num_std"],
    "rsi_strategy": ["period", "oversold", "overbought"],
    "momentum_strategy": ["lookback", "threshold"],
    "factor_strategy": ["threshold"],
    "mean_reversion": ["period", "z_entry", "z_exit"],
    "pairs_trading": ["lookback", "z_entry", "z_exit"],
    "trend_following": ["fast", "slow", "atr_period"],
    "breakout": ["period"],
    "factor_rotation": [],
    "risk_parity_dynamic": ["target_vol"],
    "carry_trade": [],
    "volatility_targeting": ["target_vol", "lookback"],
}


def count_free_params(strategy_name: str, params: dict) -> int:
    """Count number of tunable parameters in a strategy configuration.

    Uses the known parameter registry for recognized strategies.
    For unknown strategies, counts all keys in the params dict.

    Args:
        strategy_name: Name of the strategy (e.g., 'sma_crossover').
        params: Dictionary of parameter name -> value.

    Returns:
        Number of free (tunable) parameters.
    """
    if strategy_name in _STRATEGY_PARAMS:
        known = _STRATEGY_PARAMS[strategy_name]
        return len(known)
    # Fallback: count all provided params
    return len(params)


def complexity_budget(
    n_params: int,
    n_observations: int,
    min_obs_per_param: int = 100,
) -> dict:
    """Assess whether data is sufficient for model complexity.

    A model with k free parameters optimized on n observations
    needs at least min_obs_per_param * k observations to avoid
    overfitting.

    Args:
        n_params: Number of free parameters.
        n_observations: Number of data observations.
        min_obs_per_param: Minimum observations per parameter (default 100).

    Returns:
        Dictionary with keys:
            n_params: int
            n_obs: int
            ratio: float (n_obs / n_params, or inf if n_params == 0)
            warning: str (human-readable assessment)
            risk_level: str ('low', 'medium', 'high')
    """
    if n_params <= 0:
        return {
            "n_params": n_params,
            "n_obs": n_observations,
            "ratio": float("inf"),
            "warning": "No tunable parameters.",
            "risk_level": "low",
        }

    ratio = n_observations / n_params
    required = min_obs_per_param * n_params

    if ratio >= min_obs_per_param:
        risk_level = "low"
        warning = (
            f"Sufficient data: {n_observations} obs for {n_params} params "
            f"(ratio {ratio:.0f}, need {min_obs_per_param})."
        )
    elif ratio >= min_obs_per_param / 2:
        risk_level = "medium"
        warning = (
            f"Borderline data: {n_observations} obs for {n_params} params "
            f"(ratio {ratio:.0f}, recommended {min_obs_per_param}). "
            f"Need {required} total observations."
        )
    else:
        risk_level = "high"
        warning = (
            f"Insufficient data: {n_observations} obs for {n_params} params "
            f"(ratio {ratio:.0f}, recommended {min_obs_per_param}). "
            f"Need {required} total observations. High overfitting risk."
        )

    return {
        "n_params": n_params,
        "n_obs": n_observations,
        "ratio": ratio,
        "warning": warning,
        "risk_level": risk_level,
    }


def bayesian_information_criterion(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
) -> float:
    """Bayesian Information Criterion (BIC).

    BIC = -2 * LL + k * ln(n)

    Lower BIC indicates a better model, penalizing complexity more
    than AIC for larger sample sizes.

    Args:
        log_likelihood: Log-likelihood of the fitted model.
        n_params: Number of free parameters (k).
        n_obs: Number of observations (n).

    Returns:
        BIC value. Lower is better.

    Raises:
        ValueError: If n_obs < 1.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    return -2.0 * log_likelihood + n_params * np.log(n_obs)
