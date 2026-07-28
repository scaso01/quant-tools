"""Tests for cross-validation splitters."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    WalkForwardCV,
)


# =============================================================================
# PurgedKFold Tests
# =============================================================================


class TestPurgedKFold:
    """Test PurgedKFold splitter."""

    def test_correct_number_of_splits(self):
        """Should produce n_splits folds."""
        X = np.arange(100)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        splits = cv.split(X)
        assert len(splits) == 5

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must be disjoint."""
        X = np.arange(200)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        for train_idx, test_idx in cv.split(X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_embargo_creates_gap(self):
        """With embargo, there should be a gap between train and test."""
        X = np.arange(1000)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.05)
        splits = cv.split(X)
        for train_idx, test_idx in splits:
            test_min, test_max = test_idx.min(), test_idx.max()
            # Train indices near test boundaries should be excluded
            embargo_size = int(1000 * 0.05)
            for idx in train_idx:
                # No train index should be within embargo of test boundaries
                if idx < test_min:
                    assert idx < test_min - embargo_size + 1
                elif idx > test_max:
                    assert idx >= test_max + embargo_size

    def test_all_indices_covered(self):
        """Every index should appear in at least one test set (no embargo)."""
        X = np.arange(100)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        all_test = np.concatenate([test for _, test in cv.split(X)])
        assert set(all_test) == set(range(100))

    def test_invalid_n_splits_raises(self):
        """n_splits < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=1)

    def test_small_dataset(self):
        """Should handle small datasets gracefully."""
        X = np.arange(10)
        cv = PurgedKFold(n_splits=2, embargo_pct=0.0)
        splits = cv.split(X)
        assert len(splits) == 2
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# =============================================================================
# CombinatorialPurgedCV Tests
# =============================================================================


class TestCombinatorialPurgedCV:
    """Test CombinatorialPurgedCV splitter."""

    def test_correct_number_of_splits(self):
        """Should produce C(n_groups, n_test_groups) combinations."""
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
        assert cv.n_splits == 15  # C(6,2) = 15

    def test_splits_match_expected_count(self):
        """Actual splits should match n_splits property."""
        X = np.arange(600)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
        splits = cv.split(X)
        assert len(splits) == cv.n_splits

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must be disjoint."""
        X = np.arange(600)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
        for train_idx, test_idx in cv.split(X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_embargo_reduces_train_size(self):
        """Embargo should remove training samples near test boundaries."""
        X = np.arange(600)
        cv_no_embargo = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0)
        cv_embargo = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.05)
        splits_no = cv_no_embargo.split(X)
        splits_emb = cv_embargo.split(X)
        # With embargo, training sets should generally be smaller
        for (train_no, _), (train_emb, _) in zip(splits_no, splits_emb):
            assert len(train_emb) <= len(train_no)

    def test_invalid_groups_raises(self):
        """n_test_groups >= n_groups should raise ValueError."""
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=3, n_test_groups=3)

    def test_n_groups_minimum(self):
        """n_groups < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_groups must be >= 2"):
            CombinatorialPurgedCV(n_groups=1, n_test_groups=0)


# =============================================================================
# WalkForwardCV Tests
# =============================================================================


class TestWalkForwardCV:
    """Test WalkForwardCV splitter."""

    def test_correct_window_sizes(self):
        """IS and OOS windows should have correct sizes."""
        X = np.arange(500)
        cv = WalkForwardCV(is_window=200, oos_window=100)
        for train_idx, test_idx in cv.split(X):
            assert len(train_idx) == 200
            assert len(test_idx) == 100

    def test_rolling_progression(self):
        """Each split should advance by step size."""
        X = np.arange(1000)
        cv = WalkForwardCV(is_window=200, oos_window=100, step=100)
        splits = cv.split(X)
        assert len(splits) >= 2
        # First split starts at 0, second at step=100
        assert splits[0][0][0] == 0
        assert splits[1][0][0] == 100

    def test_no_overlap_between_is_and_oos(self):
        """IS and OOS should not overlap within a single split."""
        X = np.arange(500)
        cv = WalkForwardCV(is_window=200, oos_window=100)
        for train_idx, test_idx in cv.split(X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_invalid_window_raises(self):
        """Zero or negative windows should raise ValueError."""
        with pytest.raises(ValueError, match="is_window must be >= 1"):
            WalkForwardCV(is_window=0, oos_window=100)
        with pytest.raises(ValueError, match="oos_window must be >= 1"):
            WalkForwardCV(is_window=100, oos_window=0)

    def test_insufficient_data_returns_empty(self):
        """Data shorter than IS + OOS should produce no splits."""
        X = np.arange(50)
        cv = WalkForwardCV(is_window=200, oos_window=100)
        splits = cv.split(X)
        assert len(splits) == 0

    def test_custom_step_size(self):
        """Custom step size smaller than OOS creates overlapping OOS windows."""
        X = np.arange(500)
        cv = WalkForwardCV(is_window=100, oos_window=100, step=50)
        splits = cv.split(X)
        # With step=50 and total space=400 after IS, we get many splits
        assert len(splits) >= 4
