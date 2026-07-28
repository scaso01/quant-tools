"""HMM-based market regime detection using returns + volatility."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

log = logging.getLogger("quant_tools")


@dataclass(frozen=True)
class RegimeState:
    """Current market regime with probability and transitions."""

    regime_id: int
    label: str  # "bull", "bear", "high_vol", "recovery"
    probability: float
    transition_probs: dict[str, float] = field(default_factory=dict)


class RegimeDetector:
    """HMM-based market regime detection using returns + volatility.

    Uses hmmlearn.hmm.GaussianHMM (lazy import) to fit a Hidden Markov Model
    on a feature matrix of returns and rolling volatility. Labels regimes
    by mean return: highest=bull, lowest=bear, middle=high_vol.
    """

    def __init__(self, n_regimes: int = 3) -> None:
        self.n_regimes = n_regimes
        self._model = None
        self._labels: dict[int, str] = {}

    def _build_features(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build feature matrix from returns and volatility."""
        if volatility is None:
            vol_series = pd.Series(returns).rolling(20).std()
            volatility = vol_series.bfill().values
        return np.column_stack([returns, volatility])

    def fit(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> RegimeDetector:
        """Fit HMM on returns (and optionally volatility).

        Uses hmmlearn.hmm.GaussianHMM (lazy import).
        Labels regimes by mean return: highest=bull, lowest=bear,
        middle=high_vol.

        Args:
            returns: 1-D array of daily returns.
            volatility: Optional 1-D array of rolling volatility.
                If None, computed as 20-day rolling std of returns.

        Returns:
            self for method chaining.

        Raises:
            ImportError: If hmmlearn is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn not installed. pip install hmmlearn"
            )

        X = self._build_features(returns, volatility)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self._model.fit(X)

        # Label regimes by mean return (column 0)
        means = self._model.means_[:, 0]
        sorted_idx = np.argsort(means)

        if self.n_regimes == 2:
            labels = ["bear", "bull"]
        elif self.n_regimes == 4:
            labels = ["bear", "high_vol", "recovery", "bull"]
        else:
            labels = ["bear", "high_vol", "bull"]

        self._labels = {}
        for i, idx in enumerate(sorted_idx):
            self._labels[int(idx)] = labels[min(i, len(labels) - 1)]

        log.info(
            "HMM fitted with %d regimes: %s",
            self.n_regimes,
            self._labels,
        )
        return self

    def predict(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> int:
        """Predict current regime.

        Args:
            returns: 1-D array of recent returns.
            volatility: Optional volatility array.

        Returns:
            regime_id (int) for the last observation.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        X = self._build_features(returns, volatility)
        hidden_states = self._model.predict(X)
        return int(hidden_states[-1])

    def predict_proba(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict regime probabilities for the last observation.

        Args:
            returns: 1-D array of recent returns.
            volatility: Optional volatility array.

        Returns:
            Array of shape (n_regimes,) with probabilities.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        X = self._build_features(returns, volatility)
        proba = self._model.predict_proba(X)
        return proba[-1]

    def get_transition_matrix(self) -> np.ndarray:
        """Return transition probability matrix.

        Returns:
            Array of shape (n_regimes, n_regimes). Row i gives
            probabilities of transitioning from regime i to each other.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before get_transition_matrix()")
        return self._model.transmat_

    def get_current_state(
        self,
        returns: np.ndarray,
        volatility: np.ndarray | None = None,
    ) -> RegimeState:
        """Return full RegimeState for current (last) observation.

        Args:
            returns: 1-D array of recent returns.
            volatility: Optional volatility array.

        Returns:
            RegimeState with regime_id, label, probability, and
            transition probabilities to other regimes.
        """
        regime_id = self.predict(returns, volatility)
        proba = self.predict_proba(returns, volatility)
        transmat = self.get_transition_matrix()

        transition_probs = {}
        for j in range(self.n_regimes):
            target_label = self._labels.get(j, f"regime_{j}")
            transition_probs[target_label] = float(transmat[regime_id, j])

        return RegimeState(
            regime_id=regime_id,
            label=self._labels.get(regime_id, "unknown"),
            probability=float(proba[regime_id]),
            transition_probs=transition_probs,
        )

    def regime_label(self, regime_id: int) -> str:
        """Get the human-readable label for a regime id."""
        return self._labels.get(regime_id, "unknown")

    def suggest_allocation(self, regime: str) -> dict[str, float]:
        """Suggest portfolio allocation weights per regime.

        Args:
            regime: One of "bull", "bear", "high_vol", "recovery".

        Returns:
            Dict with suggested weights for equity, bond, gold.
        """
        allocations = {
            "bull": {"equity": 0.7, "bond": 0.2, "gold": 0.1},
            "bear": {"equity": 0.2, "bond": 0.5, "gold": 0.3},
            "high_vol": {"equity": 0.3, "bond": 0.4, "gold": 0.3},
            "recovery": {"equity": 0.5, "bond": 0.3, "gold": 0.2},
        }
        return allocations.get(regime, {"equity": 0.34, "bond": 0.33, "gold": 0.33})
