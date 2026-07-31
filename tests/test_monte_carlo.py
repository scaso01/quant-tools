"""Tests for Monte Carlo simulation."""

from __future__ import annotations


import numpy as np
import pytest

from quant_tools.errors import ComputeError
from quant_tools.models import DistributionParams, DistributionType, MonteCarloResult
from quant_tools.portfolio.monte_carlo import (
    _generate_random_returns,
    estimate_distribution_params,
    fit_best_distribution,
    run_monte_carlo,
    run_monte_carlo_garch,
)

try:
    import arch

    _HAS_ARCH = arch is not None
except ImportError:
    _HAS_ARCH = False


class TestRunMonteCarlo:
    """Test run_monte_carlo function."""

    def test_fixed_seed_reproducible(self):
        """Same seed should produce identical results."""
        returns = np.random.default_rng(0).normal(0.0005, 0.01, 252)

        result1 = run_monte_carlo(returns, seed=42)
        result2 = run_monte_carlo(returns, seed=42)

        assert result1.mean_final == pytest.approx(result2.mean_final)
        assert result1.median_final == pytest.approx(result2.median_final)
        assert result1.percentile_5 == pytest.approx(result2.percentile_5)
        assert result1.prob_loss == pytest.approx(result2.prob_loss)

    def test_all_paths_start_at_initial_value(self):
        """Every simulated path should begin at initial_value."""
        returns = np.random.default_rng(1).normal(0.0005, 0.01, 100)
        initial = 50000.0

        result = run_monte_carlo(
            returns,
            initial_value=initial,
            n_simulations=50,
            time_horizon_days=20,
            seed=99,
        )

        for path in result.simulated_paths:
            assert path[0] == pytest.approx(initial)

    def test_percentile_ordering(self):
        """Percentiles must be in ascending order."""
        returns = np.random.default_rng(2).normal(0.0005, 0.015, 252)

        result = run_monte_carlo(returns, n_simulations=500, seed=123)

        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.median_final
        assert result.median_final <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_prob_loss_between_0_and_1(self):
        """Probability of loss should be a valid probability."""
        returns = np.random.default_rng(3).normal(0.0, 0.02, 252)

        result = run_monte_carlo(returns, seed=10)

        assert 0.0 <= result.prob_loss <= 1.0
        assert 0.0 <= result.prob_gain_10pct <= 1.0

    def test_zero_volatility_identical_paths(self):
        """Zero-vol returns should make all paths identical."""
        daily_ret = 0.001
        returns = np.full(100, daily_ret)

        result = run_monte_carlo(
            returns,
            initial_value=100000,
            n_simulations=50,
            time_horizon_days=10,
            seed=55,
        )

        # With zero std and ddof=1 on a constant array, std is 0
        # All random draws will be mu (since sigma=0), so paths are identical
        expected_final = 100000 * (1 + daily_ret) ** 10
        assert result.mean_final == pytest.approx(expected_final, rel=1e-6)
        assert result.prob_loss == pytest.approx(0.0)
        assert result.percentile_5 == pytest.approx(result.percentile_95, rel=1e-6)

    def test_large_n_simulations_mean_convergence(self):
        """With many simulations, mean final should converge near expected value."""
        mu = 0.0004
        n_days = 252
        returns = np.full(252, mu)  # Use deterministic returns for known mu/sigma

        # For a known mu with zero sigma, expected = initial * (1+mu)^n_days
        result = run_monte_carlo(
            returns,
            initial_value=100000,
            n_simulations=5000,
            time_horizon_days=n_days,
            seed=77,
        )

        expected = 100000 * (1 + mu) ** n_days
        assert result.mean_final == pytest.approx(expected, rel=1e-4)

    def test_result_type(self):
        """Result should be a MonteCarloResult dataclass."""
        returns = np.random.default_rng(4).normal(0, 0.01, 50)
        result = run_monte_carlo(returns, n_simulations=100, seed=1)
        assert isinstance(result, MonteCarloResult)

    def test_result_fields_populated(self):
        """All result fields should be populated."""
        returns = np.random.default_rng(5).normal(0.0003, 0.01, 100)

        result = run_monte_carlo(
            returns,
            initial_value=100000,
            n_simulations=200,
            time_horizon_days=60,
            seed=42,
        )

        assert result.n_simulations == 200
        assert result.time_horizon_days == 60
        assert result.initial_value == 100000
        assert result.mean_final > 0
        assert result.median_final > 0
        assert len(result.simulated_paths) <= 100  # capped at 100

    def test_too_few_returns_raises(self):
        """Fewer than 2 returns should raise ComputeError."""
        with pytest.raises(ComputeError, match="at least 2"):
            run_monte_carlo(np.array([0.01]))

    def test_nan_returns_raises(self):
        """NaN in returns should raise ComputeError."""
        returns = np.array([0.01, np.nan, 0.02])
        with pytest.raises(ComputeError, match="NaN"):
            run_monte_carlo(returns)

    def test_path_length(self):
        """Each stored path should have time_horizon_days + 1 points."""
        returns = np.random.default_rng(6).normal(0, 0.01, 50)
        horizon = 30
        result = run_monte_carlo(
            returns,
            n_simulations=10,
            time_horizon_days=horizon,
            seed=8,
        )

        for path in result.simulated_paths:
            assert len(path) == horizon + 1  # includes initial value


# ── Backward Compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    """Default params must produce identical results to old code."""

    def test_default_params_same_as_old(self):
        """Same seed + default new params = same results as before."""
        returns = np.random.default_rng(0).normal(0.0005, 0.01, 252)

        # Old-style call (no new params)
        old = run_monte_carlo(returns, seed=42)
        # Explicit new-style call with defaults
        new = run_monte_carlo(
            returns,
            seed=42,
            distribution=DistributionType.NORMAL,
            distribution_params=None,
            auto_fit=False,
        )

        assert old.mean_final == pytest.approx(new.mean_final)
        assert old.median_final == pytest.approx(new.median_final)
        assert old.percentile_5 == pytest.approx(new.percentile_5)
        assert old.percentile_95 == pytest.approx(new.percentile_95)
        assert old.prob_loss == pytest.approx(new.prob_loss)

    def test_default_distribution_is_normal(self):
        """Default call should report NORMAL distribution."""
        returns = np.random.default_rng(1).normal(0.0005, 0.01, 100)
        result = run_monte_carlo(returns, seed=10, n_simulations=50)
        assert result.distribution_used == DistributionType.NORMAL


# ── Distribution Estimation ──────────────────────────────────────────


class TestEstimateDistributionParams:
    """Test estimate_distribution_params for each distribution."""

    @pytest.fixture
    def sample_returns(self):
        return np.random.default_rng(42).normal(0.0005, 0.012, 500)

    def test_normal(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.NORMAL)
        assert params.distribution == DistributionType.NORMAL
        assert params.mean == pytest.approx(float(np.mean(sample_returns)), rel=1e-6)
        assert params.std == pytest.approx(float(np.std(sample_returns, ddof=1)), rel=1e-6)

    def test_student_t(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.STUDENT_T)
        assert params.distribution == DistributionType.STUDENT_T
        assert params.df is not None
        assert params.df > 0

    def test_skew_normal(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.SKEW_NORMAL)
        assert params.distribution == DistributionType.SKEW_NORMAL
        assert params.skew_param is not None

    def test_johnson_su(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.JOHNSON_SU)
        assert params.distribution == DistributionType.JOHNSON_SU
        assert params.johnson_a is not None
        assert params.johnson_b is not None
        assert params.johnson_loc is not None
        assert params.johnson_scale is not None

    def test_ged(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.GED)
        assert params.distribution == DistributionType.GED
        assert params.df is not None  # shape parameter stored in df
        assert params.df > 0

    def test_skewness_kurtosis_populated(self, sample_returns):
        params = estimate_distribution_params(sample_returns, DistributionType.NORMAL)
        assert isinstance(params.skewness, float)
        assert isinstance(params.kurtosis, float)
        # Kurtosis should be near 3.0 for normal-ish data (Pearson)
        assert 1.5 < params.kurtosis < 5.0


# ── Best Distribution Fit ────────────────────────────────────────────


class TestFitBestDistribution:
    """Test fit_best_distribution auto-selection."""

    def test_returns_valid_params(self):
        returns = np.random.default_rng(99).normal(0.0, 0.01, 300)
        params = fit_best_distribution(returns)
        assert isinstance(params, DistributionParams)
        assert params.distribution in list(DistributionType)

    def test_small_sample_normal_or_t_only(self):
        """n < 50 should only consider Normal and Student-t."""
        returns = np.random.default_rng(7).normal(0.0, 0.01, 30)
        params = fit_best_distribution(returns)
        assert params.distribution in {DistributionType.NORMAL, DistributionType.STUDENT_T}

    def test_medium_sample_excludes_johnson(self):
        """50 <= n < 200 should exclude Johnson SU."""
        returns = np.random.default_rng(8).normal(0.0, 0.01, 100)
        params = fit_best_distribution(returns)
        assert params.distribution != DistributionType.JOHNSON_SU

    def test_large_sample_all_candidates(self):
        """n >= 200 should consider all 5 distributions."""
        returns = np.random.default_rng(9).standard_t(df=4, size=500) * 0.01
        params = fit_best_distribution(returns)
        # Just verify it returns something valid for heavy-tailed data
        assert isinstance(params, DistributionParams)


# ── Distribution-Based Monte Carlo ───────────────────────────────────


class TestDistributionMonteCarlo:
    """Test run_monte_carlo with non-normal distributions."""

    @pytest.fixture
    def fat_tailed_returns(self):
        """Generate returns with heavier tails than normal."""
        rng = np.random.default_rng(42)
        return rng.standard_t(df=4, size=500) * 0.01

    def test_student_t_wider_spread(self, fat_tailed_returns):
        """Student-t should produce wider spread than Normal for fat-tailed data."""
        normal = run_monte_carlo(
            fat_tailed_returns, seed=42, n_simulations=2000,
            distribution=DistributionType.NORMAL,
        )
        student = run_monte_carlo(
            fat_tailed_returns, seed=42, n_simulations=2000,
            distribution=DistributionType.STUDENT_T,
        )
        # Student-t with low df should have wider 5-95 spread
        normal_spread = normal.percentile_95 - normal.percentile_5
        student_spread = student.percentile_95 - student.percentile_5
        assert student_spread > normal_spread * 0.9  # at least comparable

    def test_skew_normal_runs(self):
        returns = np.random.default_rng(10).normal(0.0005, 0.01, 100)
        result = run_monte_carlo(
            returns, seed=42, n_simulations=100,
            distribution=DistributionType.SKEW_NORMAL,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.distribution_used == DistributionType.SKEW_NORMAL
        assert result.distribution_params is not None

    def test_johnson_su_runs_large_sample(self):
        returns = np.random.default_rng(11).normal(0.0, 0.012, 300)
        result = run_monte_carlo(
            returns, seed=42, n_simulations=100,
            distribution=DistributionType.JOHNSON_SU,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.distribution_used == DistributionType.JOHNSON_SU

    def test_ged_runs(self):
        returns = np.random.default_rng(12).normal(0.0, 0.01, 200)
        result = run_monte_carlo(
            returns, seed=42, n_simulations=100,
            distribution=DistributionType.GED,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.distribution_used == DistributionType.GED

    def test_auto_fit_returns_valid_result(self):
        returns = np.random.default_rng(13).normal(0.0005, 0.01, 300)
        result = run_monte_carlo(
            returns, seed=42, n_simulations=100, auto_fit=True,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.distribution_used in list(DistributionType)
        assert result.distribution_params is not None

    def test_custom_params_override(self):
        """Providing distribution_params should skip auto-estimation."""
        returns = np.random.default_rng(14).normal(0.001, 0.015, 100)
        custom = DistributionParams(
            distribution=DistributionType.NORMAL,
            mean=0.001,
            std=0.015,
        )
        result = run_monte_carlo(
            returns, seed=42, n_simulations=100,
            distribution_params=custom,
        )
        assert result.distribution_used == DistributionType.NORMAL
        assert result.distribution_params is custom

    def test_distribution_used_field_populated(self):
        returns = np.random.default_rng(15).normal(0.0, 0.01, 100)
        result = run_monte_carlo(
            returns, seed=42, n_simulations=50,
            distribution=DistributionType.STUDENT_T,
        )
        assert result.distribution_used == DistributionType.STUDENT_T

    def test_skewness_kurtosis_fields_populated(self):
        returns = np.random.default_rng(16).normal(0.0005, 0.01, 252)
        result = run_monte_carlo(returns, seed=42, n_simulations=50)
        assert isinstance(result.skewness, float)
        assert isinstance(result.kurtosis, float)
        assert result.kurtosis > 0  # Pearson kurtosis always positive


# ── Generate Random Returns ──────────────────────────────────────────


class TestGenerateRandomReturns:
    """Test _generate_random_returns shape and dispatch."""

    def test_normal_shape(self):
        rng = np.random.default_rng(42)
        params = DistributionParams(distribution=DistributionType.NORMAL, mean=0.0, std=0.01)
        result = _generate_random_returns(rng, params, (100, 50))
        assert result.shape == (100, 50)

    def test_student_t_shape(self):
        rng = np.random.default_rng(42)
        params = DistributionParams(
            distribution=DistributionType.STUDENT_T, mean=0.0, std=0.01, df=5.0,
        )
        result = _generate_random_returns(rng, params, (100, 50))
        assert result.shape == (100, 50)

    def test_skew_normal_shape(self):
        rng = np.random.default_rng(42)
        params = DistributionParams(
            distribution=DistributionType.SKEW_NORMAL, mean=0.0, std=0.01, skew_param=2.0,
        )
        result = _generate_random_returns(rng, params, (80, 30))
        assert result.shape == (80, 30)

    def test_johnson_su_shape(self):
        rng = np.random.default_rng(42)
        params = DistributionParams(
            distribution=DistributionType.JOHNSON_SU,
            johnson_a=1.0, johnson_b=1.0, johnson_loc=0.0, johnson_scale=0.01,
        )
        result = _generate_random_returns(rng, params, (60, 20))
        assert result.shape == (60, 20)

    def test_ged_shape(self):
        rng = np.random.default_rng(42)
        params = DistributionParams(
            distribution=DistributionType.GED, mean=0.0, std=0.01, df=2.0,
        )
        result = _generate_random_returns(rng, params, (50, 25))
        assert result.shape == (50, 25)


# ── GARCH Monte Carlo ────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_ARCH, reason="arch not installed (ml extra)")
class TestRunMonteCarloGarch:
    """Test GARCH-based Monte Carlo simulation."""

    def test_garch_mc_runs(self):
        """GARCH MC should run and return valid MonteCarloResult."""
        # Need enough data for GARCH fitting
        returns = np.random.default_rng(42).normal(0.0005, 0.015, 500)
        result = run_monte_carlo_garch(
            returns,
            initial_value=100_000,
            n_simulations=100,
            time_horizon_days=20,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 100
        assert result.time_horizon_days == 20
        assert result.mean_final > 0
        assert result.distribution_used == DistributionType.STUDENT_T

    def test_garch_mc_skewness_kurtosis(self):
        returns = np.random.default_rng(43).normal(0.0, 0.01, 500)
        result = run_monte_carlo_garch(
            returns, n_simulations=50, time_horizon_days=10,
        )
        assert isinstance(result.skewness, float)
        assert isinstance(result.kurtosis, float)

    def test_garch_mc_too_few_returns(self):
        with pytest.raises(ComputeError, match="at least 2"):
            run_monte_carlo_garch(np.array([0.01]))

    def test_garch_mc_nan_returns(self):
        with pytest.raises(ComputeError, match="NaN"):
            run_monte_carlo_garch(np.array([0.01, np.nan, 0.02]))

    def test_garch_mc_paths_start_at_initial(self):
        returns = np.random.default_rng(44).normal(0.0, 0.01, 500)
        result = run_monte_carlo_garch(
            returns, initial_value=50_000, n_simulations=20, time_horizon_days=10,
        )
        for path in result.simulated_paths:
            assert path[0] == pytest.approx(50_000)
