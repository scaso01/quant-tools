"""Tests for HMM regime detection (mock hmmlearn, synthetic data)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quant_tools.portfolio.regime import RegimeDetector, RegimeState


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns(rng) -> np.ndarray:
    """252 days of synthetic returns with regime structure."""
    # Bull: high mean, low vol (first 100 days)
    bull = rng.normal(0.001, 0.008, 100)
    # Bear: negative mean, high vol (next 80 days)
    bear = rng.normal(-0.002, 0.020, 80)
    # Recovery: slight positive, moderate vol (last 72 days)
    recovery = rng.normal(0.0005, 0.012, 72)
    return np.concatenate([bull, bear, recovery])


class _MockGaussianHMM:
    """Mock HMM that mimics hmmlearn.hmm.GaussianHMM interface."""

    def __init__(self, n_components=3, **kwargs):
        self.n_components = n_components
        self.means_ = None
        self.transmat_ = None
        self._n_obs = 0

    def fit(self, X):
        self._n_obs = len(X)
        n = self.n_components
        # Mock means: sorted so regime 0=lowest, 2=highest
        self.means_ = np.array([
            [-0.002, 0.02],   # bear: low return, high vol
            [0.0, 0.01],      # high_vol: zero return, moderate vol
            [0.001, 0.005],   # bull: high return, low vol
        ])[:n]

        # Stochastic transition matrix
        self.transmat_ = np.ones((n, n)) / n
        return self

    def predict(self, X):
        # Assign regime based on return sign
        states = np.zeros(len(X), dtype=int)
        for i, row in enumerate(X):
            if row[0] > 0.0005:
                states[i] = self.n_components - 1  # bull
            elif row[0] < -0.001:
                states[i] = 0  # bear
            else:
                states[i] = min(1, self.n_components - 1)  # high_vol
        return states

    def predict_proba(self, X):
        n = self.n_components
        proba = np.ones((len(X), n)) * 0.1 / (n - 1)
        states = self.predict(X)
        for i, s in enumerate(states):
            proba[i, s] = 0.9
        # Normalize rows
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


@pytest.fixture
def mock_hmm():
    """Patch hmmlearn so the lazy import inside fit() gets our mock."""
    mock_module = MagicMock()
    mock_module.GaussianHMM = _MockGaussianHMM
    with patch.dict(
        "sys.modules",
        {"hmmlearn": mock_module, "hmmlearn.hmm": mock_module},
    ):
        yield


# ── Tests ─────────────────────────────────────────────────────────────


class TestRegimeDetectorFit:
    def test_fit_returns_self(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        result = det.fit(synthetic_returns)
        assert result is det

    def test_fit_assigns_labels(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        labels = set(det._labels.values())
        assert "bull" in labels
        assert "bear" in labels

    def test_fit_two_regimes(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=2)
        det.fit(synthetic_returns)
        labels = set(det._labels.values())
        assert "bull" in labels
        assert "bear" in labels

    def test_fit_four_regimes(self, mock_hmm):
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.01, 252)
        det = RegimeDetector(n_regimes=4)
        # Mock needs 4 components
        det.fit(returns)
        # Should have assigned 4 labels (some may repeat at boundary)
        assert len(det._labels) >= 2


class TestRegimeDetectorPredict:
    def test_predict_returns_int(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        regime_id = det.predict(synthetic_returns)
        assert isinstance(regime_id, int)
        assert 0 <= regime_id < 3

    def test_predict_proba_shape(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        proba = det.predict_proba(synthetic_returns)
        assert proba.shape == (3,)
        assert abs(proba.sum() - 1.0) < 0.01

    def test_predict_without_fit_raises(self, mock_hmm):
        det = RegimeDetector()
        with pytest.raises(RuntimeError, match="Must call fit"):
            det.predict(np.array([0.01, -0.01]))


class TestTransitionMatrix:
    def test_transition_matrix_stochastic(self, mock_hmm, synthetic_returns):
        """Rows of transition matrix should sum to 1."""
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        tm = det.get_transition_matrix()
        assert tm.shape == (3, 3)
        for row in tm:
            assert abs(row.sum() - 1.0) < 0.01

    def test_transition_matrix_without_fit_raises(self, mock_hmm):
        det = RegimeDetector()
        with pytest.raises(RuntimeError, match="Must call fit"):
            det.get_transition_matrix()


class TestGetCurrentState:
    def test_returns_regime_state(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        state = det.get_current_state(synthetic_returns)
        assert isinstance(state, RegimeState)
        assert state.label in ("bull", "bear", "high_vol", "recovery", "unknown")
        assert 0.0 <= state.probability <= 1.0

    def test_transition_probs_in_state(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        state = det.get_current_state(synthetic_returns)
        assert len(state.transition_probs) == 3
        assert abs(sum(state.transition_probs.values()) - 1.0) < 0.01


class TestRegimeLabel:
    def test_regime_label_known(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        # At least one label should be known
        found_known = False
        for rid in range(3):
            label = det.regime_label(rid)
            if label in ("bull", "bear", "high_vol"):
                found_known = True
        assert found_known

    def test_regime_label_unknown(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        det.fit(synthetic_returns)
        assert det.regime_label(999) == "unknown"


class TestSuggestAllocation:
    def test_bull_allocation(self, mock_hmm):
        det = RegimeDetector()
        alloc = det.suggest_allocation("bull")
        assert abs(sum(alloc.values()) - 1.0) < 0.01
        assert alloc["equity"] == 0.7

    def test_bear_allocation(self, mock_hmm):
        det = RegimeDetector()
        alloc = det.suggest_allocation("bear")
        assert abs(sum(alloc.values()) - 1.0) < 0.01
        assert alloc["equity"] == 0.2
        assert alloc["bond"] == 0.5

    def test_unknown_regime_allocation(self, mock_hmm):
        det = RegimeDetector()
        alloc = det.suggest_allocation("zombie_apocalypse")
        assert abs(sum(alloc.values()) - 1.0) < 0.02

    def test_all_regimes_valid_weights(self, mock_hmm):
        det = RegimeDetector()
        for regime in ("bull", "bear", "high_vol", "recovery"):
            alloc = det.suggest_allocation(regime)
            assert abs(sum(alloc.values()) - 1.0) < 0.01
            for v in alloc.values():
                assert 0.0 <= v <= 1.0


class TestEdgeCases:
    def test_single_observation(self, mock_hmm):
        det = RegimeDetector(n_regimes=2)
        returns = np.array([0.01])
        det.fit(returns)
        regime_id = det.predict(returns)
        assert isinstance(regime_id, int)

    def test_all_same_returns(self, mock_hmm):
        det = RegimeDetector(n_regimes=2)
        returns = np.zeros(50)
        det.fit(returns)
        regime_id = det.predict(returns)
        assert isinstance(regime_id, int)

    def test_custom_volatility(self, mock_hmm, synthetic_returns):
        det = RegimeDetector(n_regimes=3)
        vol = np.abs(synthetic_returns) * 2  # custom vol proxy
        det.fit(synthetic_returns, volatility=vol)
        state = det.get_current_state(synthetic_returns, volatility=vol)
        assert isinstance(state, RegimeState)
