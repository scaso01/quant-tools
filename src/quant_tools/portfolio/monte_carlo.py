"""Monte Carlo simulation for portfolio returns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

from quant_tools.errors import ComputeError
from quant_tools.models import DistributionParams, DistributionType, MonteCarloResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def estimate_distribution_params(
    returns: NDArray[np.float64],
    distribution: DistributionType = DistributionType.NORMAL,
) -> DistributionParams:
    """Estimate distribution parameters via MLE.

    Args:
        returns: 1-D array of historical daily returns.
        distribution: Which distribution family to fit.

    Returns:
        DistributionParams with fitted parameters, skewness, and kurtosis.
    """
    data = np.asarray(returns, dtype=np.float64)
    skewness = float(sp_skew(data, bias=False))
    kurt = float(sp_kurtosis(data, bias=False, fisher=False))  # excess=False → Pearson

    if distribution == DistributionType.NORMAL:
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        return DistributionParams(
            distribution=DistributionType.NORMAL,
            mean=mu,
            std=sigma,
            skewness=skewness,
            kurtosis=kurt,
        )

    if distribution == DistributionType.STUDENT_T:
        df, loc, scale = scipy_stats.t.fit(data)
        return DistributionParams(
            distribution=DistributionType.STUDENT_T,
            mean=loc,
            std=scale,
            df=df,
            skewness=skewness,
            kurtosis=kurt,
        )

    if distribution == DistributionType.SKEW_NORMAL:
        a, loc, scale = scipy_stats.skewnorm.fit(data)
        return DistributionParams(
            distribution=DistributionType.SKEW_NORMAL,
            mean=loc,
            std=scale,
            skew_param=a,
            skewness=skewness,
            kurtosis=kurt,
        )

    if distribution == DistributionType.JOHNSON_SU:
        a, b, loc, scale = scipy_stats.johnsonsu.fit(data)
        return DistributionParams(
            distribution=DistributionType.JOHNSON_SU,
            mean=loc,
            std=scale,
            johnson_a=a,
            johnson_b=b,
            johnson_loc=loc,
            johnson_scale=scale,
            skewness=skewness,
            kurtosis=kurt,
        )

    if distribution == DistributionType.GED:
        beta, loc, scale = scipy_stats.gennorm.fit(data)
        return DistributionParams(
            distribution=DistributionType.GED,
            mean=loc,
            std=scale,
            df=beta,  # GED shape parameter stored in df field
            skewness=skewness,
            kurtosis=kurt,
        )

    msg = f"Unsupported distribution: {distribution}"
    raise ComputeError(msg)


def fit_best_distribution(
    returns: NDArray[np.float64],
) -> DistributionParams:
    """Auto-select best distribution by BIC ranking.

    Fits all eligible distributions (sample size guards):
    - n < 50: Normal, Student-t only
    - n < 200: exclude Johnson SU
    - n >= 200: all 5

    Returns:
        DistributionParams for the best-fitting distribution.
    """
    data = np.asarray(returns, dtype=np.float64)
    n = len(data)

    candidates: list[DistributionType] = [DistributionType.NORMAL, DistributionType.STUDENT_T]
    if n >= 50:
        candidates.append(DistributionType.SKEW_NORMAL)
        candidates.append(DistributionType.GED)
    if n >= 200:
        candidates.append(DistributionType.JOHNSON_SU)

    best_bic = np.inf
    best_params: DistributionParams | None = None

    for dist_type in candidates:
        params = estimate_distribution_params(data, dist_type)
        bic = _compute_bic(data, dist_type, params)
        if bic < best_bic:
            best_bic = bic
            best_params = params

    # Should never be None since we always have Normal, but satisfy type checker
    assert best_params is not None  # noqa: S101
    return best_params


def _compute_bic(
    data: NDArray[np.float64],
    dist_type: DistributionType,
    params: DistributionParams,
) -> float:
    """Compute BIC for a fitted distribution."""
    n = len(data)

    if dist_type == DistributionType.NORMAL:
        log_lik = np.sum(scipy_stats.norm.logpdf(data, loc=params.mean, scale=params.std))
        k = 2
    elif dist_type == DistributionType.STUDENT_T:
        log_lik = np.sum(
            scipy_stats.t.logpdf(data, df=params.df, loc=params.mean, scale=params.std)
        )
        k = 3
    elif dist_type == DistributionType.SKEW_NORMAL:
        log_lik = np.sum(
            scipy_stats.skewnorm.logpdf(
                data, a=params.skew_param, loc=params.mean, scale=params.std
            )
        )
        k = 3
    elif dist_type == DistributionType.JOHNSON_SU:
        log_lik = np.sum(
            scipy_stats.johnsonsu.logpdf(
                data,
                a=params.johnson_a,
                b=params.johnson_b,
                loc=params.johnson_loc,
                scale=params.johnson_scale,
            )
        )
        k = 4
    elif dist_type == DistributionType.GED:
        log_lik = np.sum(
            scipy_stats.gennorm.logpdf(data, beta=params.df, loc=params.mean, scale=params.std)
        )
        k = 3
    else:
        return np.inf

    return float(k * np.log(n) - 2.0 * log_lik)


def _generate_random_returns(
    rng: np.random.Generator,
    params: DistributionParams,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Generate random returns from a fitted distribution.

    Args:
        rng: NumPy random generator.
        params: Fitted distribution parameters.
        shape: Output shape (n_simulations, time_horizon_days).

    Returns:
        Array of random returns with the given shape.
    """
    dist = params.distribution

    if dist == DistributionType.NORMAL:
        return rng.normal(params.mean, params.std, shape)

    # For scipy distributions, convert rng to integer seed for random_state
    seed_int = int(rng.integers(0, 2**31))

    if dist == DistributionType.STUDENT_T:
        return scipy_stats.t.rvs(
            df=params.df, loc=params.mean, scale=params.std,
            size=shape, random_state=seed_int,
        )

    if dist == DistributionType.SKEW_NORMAL:
        return scipy_stats.skewnorm.rvs(
            a=params.skew_param, loc=params.mean, scale=params.std,
            size=shape, random_state=seed_int,
        )

    if dist == DistributionType.JOHNSON_SU:
        return scipy_stats.johnsonsu.rvs(
            a=params.johnson_a, b=params.johnson_b,
            loc=params.johnson_loc, scale=params.johnson_scale,
            size=shape, random_state=seed_int,
        )

    if dist == DistributionType.GED:
        return scipy_stats.gennorm.rvs(
            beta=params.df, loc=params.mean, scale=params.std,
            size=shape, random_state=seed_int,
        )

    msg = f"Unsupported distribution: {dist}"
    raise ComputeError(msg)


def run_monte_carlo_garch(
    returns: NDArray[np.float64],
    initial_value: float = 100_000,
    n_simulations: int = 1000,
    time_horizon_days: int = 252,
    seed: int | None = None,
) -> MonteCarloResult:
    """GARCH(1,1)-aware Monte Carlo with skewed Student-t residuals.

    Uses the ``arch`` library to fit a GARCH(1,1) model and generate
    volatility-clustered simulation paths.

    Args:
        returns: 1-D array of historical daily returns (as decimals, e.g. 0.01 = 1%).
        initial_value: Starting portfolio value in dollars.
        n_simulations: Number of simulation paths.
        time_horizon_days: Number of trading days to simulate.
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with GARCH-simulated paths.
    """
    try:
        from arch import arch_model  # lazy import — optional [ml] dep
    except ImportError as exc:
        raise ComputeError(
            "arch library required for GARCH Monte Carlo. Install with: pip install arch"
        ) from exc

    if len(returns) < 2:
        raise ComputeError("Need at least 2 return observations for Monte Carlo")
    if np.any(np.isnan(returns)):
        raise ComputeError("Returns array contains NaN values")

    # arch expects returns scaled to percentage (e.g. 1.0 = 1%)
    pct_returns = np.asarray(returns, dtype=np.float64) * 100.0

    am = arch_model(pct_returns, vol="Garch", p=1, q=1, dist="skewt", mean="Constant")
    res = am.fit(disp="off")

    # Simulate forward paths
    sim = res.forecast(
        horizon=time_horizon_days,
        method="simulation",
        simulations=n_simulations,
        reindex=False,
    )

    # sim.simulations.residuals shape: (n_obs, n_simulations, horizon)
    # We want the simulation paths from the last observation
    # sim.simulations.values[-1] → shape (n_simulations, horizon) — simulated returns
    sim_returns_pct = sim.simulations.values[-1]  # (n_sims, horizon)
    sim_returns = sim_returns_pct / 100.0  # back to decimals

    # Build price paths
    growth_factors = 1.0 + sim_returns
    cumulative = np.cumprod(growth_factors, axis=1)
    price_paths = initial_value * cumulative

    final_values = price_paths[:, -1]

    mean_final = float(np.mean(final_values))
    median_final = float(np.median(final_values))
    percentile_5 = float(np.percentile(final_values, 5))
    percentile_25 = float(np.percentile(final_values, 25))
    percentile_75 = float(np.percentile(final_values, 75))
    percentile_95 = float(np.percentile(final_values, 95))
    prob_loss = float(np.mean(final_values < initial_value))
    prob_gain_10pct = float(np.mean(final_values >= initial_value * 1.10))

    n_store = min(n_simulations, 100)
    stored_paths: list[list[float]] = []
    for i in range(n_store):
        path = [initial_value] + price_paths[i].tolist()
        stored_paths.append(path)

    skewness = float(sp_skew(returns, bias=False))
    kurt = float(sp_kurtosis(returns, bias=False, fisher=False))

    return MonteCarloResult(
        n_simulations=n_simulations,
        time_horizon_days=time_horizon_days,
        initial_value=initial_value,
        mean_final=mean_final,
        median_final=median_final,
        percentile_5=percentile_5,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        percentile_95=percentile_95,
        prob_loss=prob_loss,
        prob_gain_10pct=prob_gain_10pct,
        simulated_paths=stored_paths,
        distribution_used=DistributionType.STUDENT_T,
        skewness=skewness,
        kurtosis=kurt,
    )


def run_monte_carlo(
    returns: NDArray,
    initial_value: float = 100000,
    n_simulations: int = 1000,
    time_horizon_days: int = 252,
    seed: int | None = None,
    distribution: DistributionType = DistributionType.NORMAL,
    distribution_params: DistributionParams | None = None,
    auto_fit: bool = False,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on historical returns.

    Uses historical daily return statistics to generate random future
    return paths. Supports multiple distribution families.

    Args:
        returns: 1-D array of historical daily returns.
        initial_value: Starting portfolio value in dollars.
        n_simulations: Number of simulation paths to generate.
        time_horizon_days: Number of trading days to simulate forward.
        seed: Random seed for reproducibility. None for random.
        distribution: Distribution type for random return generation.
        distribution_params: Pre-fitted params (skips estimation if provided).
        auto_fit: If True, auto-select best distribution via BIC.

    Returns:
        MonteCarloResult dataclass with simulation statistics.

    Raises:
        ComputeError: If returns array is too short or contains NaN.
    """
    if len(returns) < 2:
        raise ComputeError("Need at least 2 return observations for Monte Carlo")

    if np.any(np.isnan(returns)):
        raise ComputeError("Returns array contains NaN values")

    rng = np.random.default_rng(seed)

    # Determine distribution params
    if distribution_params is not None:
        params = distribution_params
    elif auto_fit:
        params = fit_best_distribution(returns)
    elif distribution == DistributionType.NORMAL:
        # Default path: manual mu/sigma for exact backward compatibility
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        params = DistributionParams(
            distribution=DistributionType.NORMAL,
            mean=mu,
            std=sigma,
        )
    else:
        params = estimate_distribution_params(returns, distribution)

    # Generate random daily returns: shape (n_simulations, time_horizon_days)
    shape = (n_simulations, time_horizon_days)
    if params.distribution == DistributionType.NORMAL:
        # Use rng.normal directly for backward compat (identical RNG sequence)
        random_returns = rng.normal(params.mean, params.std, shape)
    else:
        random_returns = _generate_random_returns(rng, params, shape)

    # Build price paths from initial value
    # cumulative product along time axis
    growth_factors = 1 + random_returns
    cumulative = np.cumprod(growth_factors, axis=1)
    price_paths = initial_value * cumulative

    # Final values for each simulation
    final_values = price_paths[:, -1]

    # Statistics
    mean_final = float(np.mean(final_values))
    median_final = float(np.median(final_values))

    percentile_5 = float(np.percentile(final_values, 5))
    percentile_25 = float(np.percentile(final_values, 25))
    percentile_75 = float(np.percentile(final_values, 75))
    percentile_95 = float(np.percentile(final_values, 95))

    # Probability of loss (final value < initial value)
    prob_loss = float(np.mean(final_values < initial_value))

    # Probability of >= 10% gain
    prob_gain_10pct = float(np.mean(final_values >= initial_value * 1.10))

    # Store a subset of paths for plotting (cap at 100 to keep memory bounded)
    n_store = min(n_simulations, 100)
    stored_paths: list[list[float]] = []
    for i in range(n_store):
        path = [initial_value] + price_paths[i].tolist()
        stored_paths.append(path)

    # Compute skewness/kurtosis of input returns
    skewness = float(sp_skew(returns, bias=False))
    kurt = float(sp_kurtosis(returns, bias=False, fisher=False))

    return MonteCarloResult(
        n_simulations=n_simulations,
        time_horizon_days=time_horizon_days,
        initial_value=initial_value,
        mean_final=mean_final,
        median_final=median_final,
        percentile_5=percentile_5,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        percentile_95=percentile_95,
        prob_loss=prob_loss,
        prob_gain_10pct=prob_gain_10pct,
        simulated_paths=stored_paths,
        distribution_used=params.distribution,
        distribution_params=params,
        skewness=skewness,
        kurtosis=kurt,
    )
