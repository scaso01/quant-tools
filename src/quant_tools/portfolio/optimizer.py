"""Portfolio optimization using cvxpy (convex) and scipy (risk parity)."""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from quant_tools.config import cfg
from quant_tools.models import (
    EfficientFrontier,
    EfficientFrontierPoint,
    OptimizationConstraints,
    PortfolioOptimization,
)

log = logging.getLogger("quant_tools")


class PortfolioOptimizer:
    """Mean-variance and alternative portfolio optimization."""

    def __init__(self, risk_free_rate: float | None = None) -> None:
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else cfg.risk_free_rate

    def optimize(self, returns_df, method: str = "markowitz") -> PortfolioOptimization:
        """Dispatch to optimization method."""
        methods = {
            "markowitz": self.optimize_markowitz,
            "risk_parity": self.optimize_risk_parity,
            "min_variance": self.optimize_min_variance,
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods)}")
        return methods[method](returns_df)

    # ── Covariance helpers ───────────────────────────────────────────

    @staticmethod
    def _build_covariance(
        returns_df,
        correlation_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build annualized covariance matrix, optionally from correlation override.

        If correlation_override is provided, combine it with historical
        volatilities and enforce PSD via eigenvalue flooring.
        """
        if correlation_override is not None:
            vols = np.std(returns_df.values, axis=0, ddof=1) * np.sqrt(252)
            D = np.diag(vols)
            Sigma = D @ correlation_override @ D
            # Enforce PSD via eigenvalue flooring
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, 1e-8)
            Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return Sigma
        return returns_df.cov().values * 252

    # ── Constraint builder ───────────────────────────────────────────

    @staticmethod
    def _build_cvxpy_constraints(
        w: cp.Variable,
        n_assets: int,
        symbols: list[str],
        constraints: OptimizationConstraints | None,
        default_min: float = 0.0,
        default_max: float = 1.0,
    ) -> list:
        """Build cvxpy constraint list from OptimizationConstraints.

        Always includes: sum(w) == 1.
        Per-asset bounds from asset_constraints (or defaults).
        Group constraints cap sector/group weight sums.
        Turnover constraint limits L1 deviation from current weights.
        """
        cvxpy_constraints: list = [cp.sum(w) == 1]

        # Ensure default bounds are feasible with sum==1
        if n_assets * default_max < 1.0:
            default_max = 1.0 / n_assets + 0.01

        # Per-asset bounds
        if constraints and constraints.asset_constraints:
            asset_map = {ac.symbol: ac for ac in constraints.asset_constraints}
            for i, sym in enumerate(symbols):
                ac = asset_map.get(sym)
                lo = ac.min_weight if ac else default_min
                hi = ac.max_weight if ac else default_max
                cvxpy_constraints.append(w[i] >= lo)
                cvxpy_constraints.append(w[i] <= hi)
        else:
            cvxpy_constraints.append(w >= default_min)
            cvxpy_constraints.append(w <= default_max)

        if constraints is None:
            return cvxpy_constraints

        # Group/sector constraints
        for gc in constraints.group_constraints:
            indices = [symbols.index(s) for s in gc.symbols if s in symbols]
            if indices:
                cvxpy_constraints.append(cp.sum(w[indices]) >= gc.min_weight)
                cvxpy_constraints.append(cp.sum(w[indices]) <= gc.max_weight)

        # Turnover constraint
        if constraints.max_turnover is not None and constraints.current_weights:
            w_old = np.array([constraints.current_weights.get(s, 0.0) for s in symbols])
            cvxpy_constraints.append(cp.norm1(w - w_old) <= constraints.max_turnover)

        return cvxpy_constraints

    # ── Result builder ───────────────────────────────────────────────

    def _make_result(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        Sigma: np.ndarray,
        assets: list[str],
        method: str,
        constraints: OptimizationConstraints | None = None,
        efficient_frontier: EfficientFrontier | None = None,
        weight_threshold: float = 0.001,
    ) -> PortfolioOptimization:
        """Build PortfolioOptimization from solved weights."""
        port_return = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (port_return - self.risk_free_rate) / (port_vol + 1e-10)

        # Compute tracking error if benchmark provided
        te = None
        if constraints and constraints.tracking_error_target is not None and constraints.benchmark_returns is not None:
            bench = np.asarray(constraints.benchmark_returns)
            if len(bench) == len(assets):
                diff = weights - bench
                te = float(np.sqrt(diff @ Sigma @ diff))

        weight_dict = {
            assets[i]: float(round(weights[i], 4))
            for i in range(len(assets))
            if weights[i] > weight_threshold
        }

        return PortfolioOptimization(
            weights=weight_dict,
            expected_return=round(port_return, 4),
            volatility=round(port_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            method=method,
            constraints_used=constraints,
            efficient_frontier=efficient_frontier,
            tracking_error=round(te, 6) if te is not None else None,
        )

    # ── Min Variance (cvxpy) ─────────────────────────────────────────

    def optimize_min_variance(
        self,
        returns_df,
        constraints: OptimizationConstraints | None = None,
    ) -> PortfolioOptimization:
        """Global minimum variance portfolio via cvxpy.

        Default bounds: [0, 0.50] per asset (matching original scipy version).
        """
        if len(returns_df.columns) == 0:
            raise ValueError("optimize_min_variance requires at least one asset")
        n_assets = len(returns_df.columns)
        assets = list(returns_df.columns)
        mu = returns_df.mean().values * 252

        corr_override = constraints.correlation_override if constraints else None
        Sigma = self._build_covariance(returns_df, corr_override)

        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        cvxpy_constraints = self._build_cvxpy_constraints(
            w, n_assets, assets, constraints,
            default_min=0.0, default_max=0.5,
        )

        prob = cp.Problem(objective, cvxpy_constraints)
        prob.solve(solver=cp.SCS, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            log.warning("Min-variance optimization %s, using equal weights", prob.status)
            weights = np.ones(n_assets) / n_assets
        else:
            weights = np.clip(w.value, 0, None)
            weights = weights / weights.sum()

        return self._make_result(weights, mu, Sigma, assets, "min_variance", constraints)

    # ── Markowitz Max-Sharpe (cvxpy parametric) ──────────────────────

    def optimize_markowitz(
        self,
        returns_df,
        constraints: OptimizationConstraints | None = None,
    ) -> PortfolioOptimization:
        """Maximum Sharpe ratio portfolio via parametric return sweep.

        Solves min-variance at N target returns and picks the point
        with the highest Sharpe ratio.

        Default bounds: [0, 0.30] per asset (matching original scipy version).
        """
        if len(returns_df.columns) == 0:
            raise ValueError("optimize_markowitz requires at least one asset")
        n_assets = len(returns_df.columns)
        assets = list(returns_df.columns)
        mu = returns_df.mean().values * 252

        corr_override = constraints.correlation_override if constraints else None
        Sigma = self._build_covariance(returns_df, corr_override)

        w = cp.Variable(n_assets)
        base_constraints = self._build_cvxpy_constraints(
            w, n_assets, assets, constraints,
            default_min=0.0, default_max=0.3,
        )

        # Find feasible return range by optimizing min/max return
        prob_min = cp.Problem(cp.Minimize(mu @ w), base_constraints)
        prob_min.solve(solver=cp.SCS, warm_start=True)
        if prob_min.status not in ("optimal", "optimal_inaccurate"):
            log.warning("Markowitz min-return probe %s, using equal weights", prob_min.status)
            weights = np.ones(n_assets) / n_assets
            return self._make_result(weights, mu, Sigma, assets, "markowitz", constraints)
        ret_min = float(mu @ w.value)

        prob_max = cp.Problem(cp.Maximize(mu @ w), base_constraints)
        prob_max.solve(solver=cp.SCS, warm_start=True)
        if prob_max.status not in ("optimal", "optimal_inaccurate"):
            log.warning("Markowitz max-return probe %s, using equal weights", prob_max.status)
            weights = np.ones(n_assets) / n_assets
            return self._make_result(weights, mu, Sigma, assets, "markowitz", constraints)
        ret_max = float(mu @ w.value)

        # Sweep target returns to find max-Sharpe point
        n_sweep = 50
        target_returns = np.linspace(ret_min, ret_max, n_sweep)
        best_sharpe = -np.inf
        best_weights = np.ones(n_assets) / n_assets

        target_param = cp.Parameter()
        sweep_constraints = base_constraints + [mu @ w >= target_param]
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        prob_sweep = cp.Problem(objective, sweep_constraints)

        for target_ret in target_returns:
            target_param.value = target_ret
            prob_sweep.solve(solver=cp.SCS, warm_start=True)
            if prob_sweep.status not in ("optimal", "optimal_inaccurate"):
                continue
            wv = np.clip(w.value, 0, None)
            wv = wv / wv.sum()
            p_ret = float(wv @ mu)
            p_vol = float(np.sqrt(wv @ Sigma @ wv))
            sharpe = (p_ret - self.risk_free_rate) / (p_vol + 1e-10)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = wv.copy()

        return self._make_result(best_weights, mu, Sigma, assets, "markowitz", constraints)

    # ── Risk Parity (scipy — non-convex) ─────────────────────────────

    def optimize_risk_parity(
        self,
        returns_df,
        constraints: OptimizationConstraints | None = None,
    ) -> PortfolioOptimization:
        """Equal risk contribution portfolio.

        Uses scipy for the non-convex risk-budget objective.
        Default bounds: [0.01, 0.50] per asset (matching original scipy version).
        """
        n_assets = len(returns_df.columns)
        assets = list(returns_df.columns)
        mu = returns_df.mean().values * 252

        corr_override = constraints.correlation_override if constraints else None
        Sigma = self._build_covariance(returns_df, corr_override)

        def risk_budget_objective(weights):
            port_vol = np.sqrt(weights @ Sigma @ weights)
            marginal_contrib = Sigma @ weights
            risk_contrib = weights * marginal_contrib / (port_vol + 1e-10)
            target = port_vol / n_assets
            return np.sum((risk_contrib - target) ** 2)

        # Build scipy bounds from constraints
        if constraints and constraints.asset_constraints:
            asset_map = {ac.symbol: ac for ac in constraints.asset_constraints}
            bounds = []
            for sym in assets:
                ac = asset_map.get(sym)
                lo = ac.min_weight if ac else 0.01
                hi = ac.max_weight if ac else 0.50
                bounds.append((lo, hi))
        else:
            bounds = [(0.01, 0.5)] * n_assets

        scipy_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Add group constraints via scipy
        if constraints and constraints.group_constraints:
            for gc in constraints.group_constraints:
                indices = [assets.index(s) for s in gc.symbols if s in assets]
                if indices:
                    scipy_constraints.append({
                        "type": "ineq",
                        "fun": lambda w, idx=indices, mx=gc.max_weight: mx - sum(w[i] for i in idx),
                    })
                    scipy_constraints.append({
                        "type": "ineq",
                        "fun": lambda w, idx=indices, mn=gc.min_weight: sum(w[i] for i in idx) - mn,
                    })

        # Turnover constraint via scipy
        if constraints and constraints.max_turnover is not None and constraints.current_weights:
            w_old = np.array([constraints.current_weights.get(s, 0.0) for s in assets])
            max_to = constraints.max_turnover
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w, wo=w_old, mt=max_to: mt - np.sum(np.abs(w - wo)),
            })

        x0 = np.ones(n_assets) / n_assets
        result = minimize(risk_budget_objective, x0, method="SLSQP", bounds=bounds, constraints=scipy_constraints)
        weights = result.x

        return self._make_result(weights, mu, Sigma, assets, "risk_parity", constraints, weight_threshold=0.0)

    # ── Efficient Frontier (cvxpy parametric sweep) ──────────────────

    def optimize_efficient_frontier(
        self,
        returns_df,
        n_points: int = 50,
        constraints: OptimizationConstraints | None = None,
    ) -> EfficientFrontier:
        """Compute efficient frontier via parametric return sweep.

        Finds min/max feasible return, then solves min-variance at N
        target returns. Identifies max-Sharpe and min-variance points.
        Uses cvxpy warm-starting for speed.
        """
        n_assets = len(returns_df.columns)
        assets = list(returns_df.columns)
        mu = returns_df.mean().values * 252

        corr_override = constraints.correlation_override if constraints else None
        Sigma = self._build_covariance(returns_df, corr_override)

        w = cp.Variable(n_assets)
        base_constraints = self._build_cvxpy_constraints(
            w, n_assets, assets, constraints,
            default_min=0.0, default_max=0.5,
        )

        # Find feasible return range
        prob_min = cp.Problem(cp.Minimize(mu @ w), base_constraints)
        prob_min.solve(solver=cp.SCS, warm_start=True)
        ret_min = float(mu @ w.value) if prob_min.status in ("optimal", "optimal_inaccurate") else float(mu.min())

        prob_max = cp.Problem(cp.Maximize(mu @ w), base_constraints)
        prob_max.solve(solver=cp.SCS, warm_start=True)
        ret_max = float(mu @ w.value) if prob_max.status in ("optimal", "optimal_inaccurate") else float(mu.max())

        # Parametric sweep
        target_returns = np.linspace(ret_min, ret_max, n_points)
        target_param = cp.Parameter()
        sweep_constraints = base_constraints + [mu @ w >= target_param]
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        prob_sweep = cp.Problem(objective, sweep_constraints)

        points: list[EfficientFrontierPoint] = []
        best_sharpe = -np.inf
        best_sharpe_point: EfficientFrontierPoint | None = None
        min_vol = np.inf
        min_vol_point: EfficientFrontierPoint | None = None

        for target_ret in target_returns:
            target_param.value = target_ret
            prob_sweep.solve(solver=cp.SCS, warm_start=True)
            if prob_sweep.status not in ("optimal", "optimal_inaccurate"):
                continue

            wv = np.clip(w.value, 0, None)
            wv = wv / wv.sum()

            p_ret = float(wv @ mu)
            p_vol = float(np.sqrt(wv @ Sigma @ wv))
            sharpe = (p_ret - self.risk_free_rate) / (p_vol + 1e-10)

            weight_dict = {
                assets[i]: float(round(wv[i], 4))
                for i in range(n_assets) if wv[i] > 0.001
            }

            point = EfficientFrontierPoint(
                expected_return=round(p_ret, 4),
                volatility=round(p_vol, 4),
                sharpe_ratio=round(sharpe, 4),
                weights=weight_dict,
            )
            points.append(point)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sharpe_point = point

            if p_vol < min_vol:
                min_vol = p_vol
                min_vol_point = point

        return EfficientFrontier(
            points=points,
            max_sharpe_point=best_sharpe_point,
            min_variance_point=min_vol_point,
        )
