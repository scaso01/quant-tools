"""Anti-overfitting metrics for backtest validation.

Provides:
- Deflated Sharpe Ratio (DSR): Adjusts Sharpe for multiple testing and non-normality.
- Probability of Backtest Overfitting (PBO): CPCV-based overfitting estimation.
- Traffic light classification for PBO risk levels.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
    sharpe_benchmark: float = 0.0,
) -> float:
    """Bailey & Lopez de Prado Deflated Sharpe Ratio.

    Adjusts for multiple testing bias and non-normality of returns.
    Tests the hypothesis that the observed Sharpe ratio is significantly
    higher than what would be expected by chance from n_trials strategies.

    Args:
        sharpe: Observed (annualized) Sharpe ratio.
        n_trials: Number of strategy configurations tested.
        skewness: Skewness of strategy returns (0 for normal).
        kurtosis: Kurtosis of strategy returns (3 for normal).
        n_obs: Number of return observations.
        sharpe_benchmark: Benchmark Sharpe to beat (default 0).

    Returns:
        p-value in [0, 1]. Lower values indicate the Sharpe is more likely
        genuine (not a product of luck/multiple testing).
    """
    if n_trials < 1:
        return 1.0
    if n_obs < 2:
        return 1.0

    # Expected maximum Sharpe from n_trials under the null hypothesis
    # (Euler-Mascheroni approximation for expected max of n_trials iid normals)
    log_trials = np.log(max(n_trials, 1))
    if log_trials < 1e-10:
        e_max_sr = sharpe_benchmark
    else:
        e_max_sr = sharpe_benchmark + np.sqrt(2 * log_trials) * (
            1 - np.euler_gamma / np.sqrt(2 * log_trials)
        )

    # Variance of the Sharpe ratio estimator (Lo, 2002)
    # Accounts for skewness and excess kurtosis of the return distribution
    var_sr = (
        1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2
    ) / (n_obs - 1)

    # Test statistic: z-score of observed Sharpe vs expected max
    z = (sharpe - e_max_sr) / np.sqrt(max(var_sr, 1e-10))

    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_groups: int = 10,
) -> float:
    """Estimate Probability of Backtest Overfitting (PBO) via CPCV.

    Splits the returns matrix into n_groups temporal blocks, then for each
    C(n_groups, n_groups//2) combination:
      1. Compute in-sample (IS) Sharpe for each strategy.
      2. Pick the IS-optimal strategy.
      3. Compute that strategy's out-of-sample (OOS) Sharpe.
      4. Check if it underperforms the OOS median.

    PBO = fraction of combos where the IS-optimal strategy underperforms
    the OOS median. Lower PBO = less overfitting.

    Args:
        returns_matrix: Shape (n_observations, n_strategies).
            Each column is a strategy's return series.
        n_groups: Number of temporal groups to split data into.

    Returns:
        PBO in [0, 1]. Lower = less overfitting.
    """
    n_obs, n_strategies = returns_matrix.shape

    if n_strategies < 2:
        return 0.0
    if n_obs < n_groups:
        return 1.0

    group_size = n_obs // n_groups
    n_test_groups = n_groups // 2

    # Count how often the IS-optimal strategy underperforms OOS median
    underperform_count = 0
    total_combos = 0

    for test_groups in combinations(range(n_groups), n_test_groups):
        test_set = set(test_groups)
        train_groups = [g for g in range(n_groups) if g not in test_set]

        # Build IS and OOS index arrays
        is_indices: list[int] = []
        oos_indices: list[int] = []
        for g in train_groups:
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else n_obs
            is_indices.extend(range(start, end))
        for g in test_groups:
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else n_obs
            oos_indices.extend(range(start, end))

        if not is_indices or not oos_indices:
            continue

        # Compute IS Sharpe for each strategy
        is_returns = returns_matrix[is_indices, :]
        is_means = np.mean(is_returns, axis=0)
        is_stds = np.std(is_returns, axis=0, ddof=1)
        is_stds = np.where(is_stds < 1e-12, 1e-12, is_stds)
        is_sharpes = is_means / is_stds

        # Pick IS-optimal strategy
        best_is_idx = int(np.argmax(is_sharpes))

        # Compute OOS Sharpe for all strategies
        oos_returns = returns_matrix[oos_indices, :]
        oos_means = np.mean(oos_returns, axis=0)
        oos_stds = np.std(oos_returns, axis=0, ddof=1)
        oos_stds = np.where(oos_stds < 1e-12, 1e-12, oos_stds)
        oos_sharpes = oos_means / oos_stds

        # Does the IS-optimal strategy underperform the OOS median?
        oos_median = float(np.median(oos_sharpes))
        if oos_sharpes[best_is_idx] < oos_median:
            underperform_count += 1

        total_combos += 1

    if total_combos == 0:
        return 1.0

    return underperform_count / total_combos


def overfitting_traffic_light(pbo: float) -> str:
    """Map PBO value to a risk level.

    Args:
        pbo: Probability of backtest overfitting in [0, 1].

    Returns:
        Risk level: 'green' (< 0.25), 'yellow' (0.25-0.50), or 'red' (>= 0.50).
    """
    if pbo < 0.25:
        return "green"
    elif pbo < 0.50:
        return "yellow"
    return "red"
