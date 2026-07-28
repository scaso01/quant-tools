"""Black-Litterman portfolio allocation with market-implied returns + views."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("quant_tools")


class BlackLitterman:
    """Black-Litterman portfolio allocation.

    Combines market-implied equilibrium returns (reverse-optimized from
    market capitalizations) with investor views to produce posterior
    expected returns and optimal portfolio weights.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        tau: float = 0.05,
    ) -> None:
        """Initialize Black-Litterman model.

        Args:
            risk_free_rate: Annual risk-free rate (default 4%).
            tau: Scalar indicating uncertainty of the prior
                (typically 0.01 to 0.10, default 0.05).
        """
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self._market_caps: dict[str, float] = {}
        self._views: list[dict] = []
        self._returns_df: pd.DataFrame | None = None

    def set_market_data(
        self,
        returns: pd.DataFrame,
        market_caps: dict[str, float],
    ) -> BlackLitterman:
        """Set historical returns and market capitalizations.

        Args:
            returns: DataFrame with columns = asset symbols, rows = daily returns.
            market_caps: Dict of {symbol: market_cap_in_dollars}.

        Returns:
            self for method chaining.
        """
        self._returns_df = returns
        self._market_caps = market_caps
        return self

    def add_absolute_view(
        self,
        asset: str,
        expected_return: float,
        confidence: float = 0.5,
    ) -> BlackLitterman:
        """Add an absolute view on an asset's expected return.

        Example: 'AAPL will return 15%' with confidence 0.7.

        Args:
            asset: Symbol string.
            expected_return: Expected annual return (e.g. 0.15 for 15%).
            confidence: View confidence [0, 1]. Higher = more weight on view.

        Returns:
            self for method chaining.
        """
        self._views.append({
            "type": "absolute",
            "asset": asset,
            "return": expected_return,
            "confidence": confidence,
        })
        return self

    def add_relative_view(
        self,
        long_asset: str,
        short_asset: str,
        spread: float,
        confidence: float = 0.5,
    ) -> BlackLitterman:
        """Add a relative view between two assets.

        Example: 'AAPL will outperform MSFT by 3%' with confidence 0.6.

        Args:
            long_asset: Symbol expected to outperform.
            short_asset: Symbol expected to underperform.
            spread: Expected return spread (e.g. 0.03 for 3%).
            confidence: View confidence [0, 1].

        Returns:
            self for method chaining.
        """
        self._views.append({
            "type": "relative",
            "long": long_asset,
            "short": short_asset,
            "spread": spread,
            "confidence": confidence,
        })
        return self

    def _get_sigma(self) -> np.ndarray:
        """Annualized covariance matrix from returns."""
        if self._returns_df is None:
            raise RuntimeError("Must call set_market_data() first")
        return self._returns_df.cov().values * 252

    def _get_symbols(self) -> list[str]:
        """Asset symbol list from returns DataFrame."""
        if self._returns_df is None:
            raise RuntimeError("Must call set_market_data() first")
        return list(self._returns_df.columns)

    def _market_weights(self) -> np.ndarray:
        """Market-cap-weighted portfolio weights."""
        symbols = self._get_symbols()
        caps = np.array([self._market_caps.get(s, 0.0) for s in symbols])
        total = caps.sum()
        if total <= 0:
            n = len(symbols)
            return np.ones(n) / n
        return caps / total

    def equilibrium_returns(self) -> pd.Series:
        """Market-implied equilibrium returns (reverse optimization from market caps).

        pi = delta * Sigma * w_mkt

        where delta = (market_return - rf) / market_variance.
        Uses a default delta of 2.5 (standard assumption) if market return
        cannot be computed.

        Returns:
            pd.Series of annual equilibrium returns indexed by symbol.
        """
        symbols = self._get_symbols()
        sigma = self._get_sigma()
        w_mkt = self._market_weights()

        # Risk aversion coefficient (delta)
        # delta = (E[R_m] - rf) / sigma_m^2
        market_var = float(w_mkt @ sigma @ w_mkt)
        if market_var > 0:
            # Use historical market return as proxy
            if self._returns_df is not None:
                mkt_return = float(
                    (self._returns_df.values @ w_mkt).mean() * 252
                )
                delta = (mkt_return - self.risk_free_rate) / market_var
                delta = max(delta, 0.5)  # Floor at 0.5
            else:
                delta = 2.5
        else:
            delta = 2.5

        # Equilibrium returns: pi = delta * Sigma * w_mkt
        pi = delta * sigma @ w_mkt

        return pd.Series(pi, index=symbols)

    def _build_views_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (pick matrix), Q (view returns), Omega (uncertainty).

        Returns:
            Tuple of (P, Q, Omega) arrays.
        """
        symbols = self._get_symbols()
        sigma = self._get_sigma()
        n = len(symbols)

        if not self._views:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P_rows = []
        Q_vals = []
        confidences = []

        for view in self._views:
            if view["type"] == "absolute":
                idx = symbols.index(view["asset"]) if view["asset"] in symbols else -1
                if idx < 0:
                    continue
                row = np.zeros(n)
                row[idx] = 1.0
                P_rows.append(row)
                Q_vals.append(view["return"])
                confidences.append(view["confidence"])

            elif view["type"] == "relative":
                long_idx = (
                    symbols.index(view["long"]) if view["long"] in symbols else -1
                )
                short_idx = (
                    symbols.index(view["short"]) if view["short"] in symbols else -1
                )
                if long_idx < 0 or short_idx < 0:
                    continue
                row = np.zeros(n)
                row[long_idx] = 1.0
                row[short_idx] = -1.0
                P_rows.append(row)
                Q_vals.append(view["spread"])
                confidences.append(view["confidence"])

        if not P_rows:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P = np.array(P_rows)
        Q = np.array(Q_vals)

        # Omega: diagonal uncertainty matrix
        # Higher confidence = lower uncertainty
        # omega_i = (1/c_i - 1) * p_i' * tau * Sigma * p_i
        k = len(P_rows)
        omega_diag = np.zeros(k)
        for i in range(k):
            c = max(confidences[i], 0.01)  # Floor to avoid division by zero
            view_var = float(P[i] @ (self.tau * sigma) @ P[i])
            omega_diag[i] = (1.0 / c - 1.0) * view_var

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def posterior_returns(self) -> pd.Series:
        """Posterior returns after incorporating views.

        BL formula:
        mu_bl = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

        If no views are set, returns equilibrium returns.

        Returns:
            pd.Series of posterior annual returns indexed by symbol.
        """
        symbols = self._get_symbols()
        sigma = self._get_sigma()
        pi = self.equilibrium_returns().values

        P, Q, Omega = self._build_views_matrices()

        if len(P) == 0:
            return pd.Series(pi, index=symbols)

        tau_sigma = self.tau * sigma
        tau_sigma_inv = np.linalg.pinv(tau_sigma)

        # Omega inverse
        omega_inv = np.linalg.pinv(Omega)

        # Posterior precision
        precision = tau_sigma_inv + P.T @ omega_inv @ P

        # Posterior mean
        posterior_cov = np.linalg.pinv(precision)
        posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        return pd.Series(posterior_mean, index=symbols)

    def optimize(self) -> dict[str, float]:
        """Optimize portfolio using posterior returns.

        Uses mean-variance optimization: w* = Sigma^-1 * mu_posterior
        (then normalized to sum to 1 with non-negativity).

        Returns:
            Dict of {symbol: weight}, weights sum to 1.0.
        """
        symbols = self._get_symbols()
        sigma = self._get_sigma()
        mu_post = self.posterior_returns().values

        # Unconstrained optimal: w* proportional to Sigma^-1 * mu
        sigma_inv = np.linalg.pinv(sigma)
        raw_weights = sigma_inv @ mu_post

        # Apply non-negativity (long-only)
        raw_weights = np.maximum(raw_weights, 0.0)

        # Normalize
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights / total
        else:
            weights = np.ones(len(symbols)) / len(symbols)

        return {
            symbols[i]: round(float(weights[i]), 6)
            for i in range(len(symbols))
            if weights[i] > 1e-6
        }
