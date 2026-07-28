"""Cross-validation splitters for time-series backtesting.

Provides three CV strategies designed for financial data:
- PurgedKFold: K-fold with purging + embargo to prevent information leakage.
- CombinatorialPurgedCV: De Prado's CPCV for all C(n,k) train/test combos.
- WalkForwardCV: Rolling in-sample / out-of-sample windows.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


class PurgedKFold:
    """Time-series K-fold with purging and embargo.

    Purging removes train samples that temporally overlap with the test period.
    Embargo adds a gap after the test set before training data resumes,
    preventing information leakage from serial correlation.

    Args:
        n_splits: Number of folds.
        embargo_pct: Fraction of total samples to use as embargo gap.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index pairs.

        Args:
            X: Array of shape (n_samples, ...). Only length is used.
            y: Ignored. Present for API compatibility.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(X)
        embargo = int(n * self.embargo_pct)
        fold_size = n // self.n_splits
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            test_idx = np.arange(test_start, test_end)

            # Train = everything before (test_start - embargo) AND after (test_end + embargo)
            train_before = np.arange(0, max(0, test_start - embargo))
            train_after = np.arange(min(n, test_end + embargo), n)
            train_idx = np.concatenate([train_before, train_after])

            if len(train_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits


class CombinatorialPurgedCV:
    """De Prado's Combinatorial Purged Cross-Validation.

    Generates all C(n_groups, n_test_groups) unique train/test combinations
    from time-series data. Each combination uses n_test_groups as the test set
    and the remaining groups (with purging and embargo) as the training set.

    This produces far more paths than standard K-fold, enabling robust
    estimation of strategy performance distribution and overfitting probability.

    Args:
        n_groups: Number of groups to divide data into.
        n_test_groups: Number of groups to use as test set per split.
        embargo_pct: Fraction of total samples to use as embargo gap.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups ({n_test_groups}) must be < n_groups ({n_groups})"
            )
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate all combinatorial splits with purging and embargo.

        Args:
            X: Array of shape (n_samples, ...). Only length is used.
            y: Ignored. Present for API compatibility.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(X)
        embargo = int(n * self.embargo_pct)
        group_size = n // self.n_groups

        # Build group boundaries
        group_ranges: list[tuple[int, int]] = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n
            group_ranges.append((start, end))

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        # All C(n_groups, n_test_groups) test group combinations
        for test_group_combo in combinations(range(self.n_groups), self.n_test_groups):
            test_set = set(test_group_combo)

            # Collect test indices
            test_indices: list[int] = []
            for g in test_group_combo:
                start, end = group_ranges[g]
                test_indices.extend(range(start, end))
            test_idx = np.array(test_indices, dtype=np.intp)

            # Collect train indices: groups not in test set, with embargo
            train_indices: list[int] = []
            for g in range(self.n_groups):
                if g in test_set:
                    continue
                start, end = group_ranges[g]
                for idx in range(start, end):
                    # Check embargo: index must not be within 'embargo' samples
                    # of any test group boundary
                    too_close = False
                    for tg in test_group_combo:
                        tg_start, tg_end = group_ranges[tg]
                        if (tg_start - embargo) <= idx < (tg_end + embargo):
                            too_close = True
                            break
                    if not too_close:
                        train_indices.append(idx)

            train_idx = np.array(train_indices, dtype=np.intp) if train_indices else np.array([], dtype=np.intp)

            if len(train_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    @property
    def n_splits(self) -> int:
        """Number of unique train/test combinations."""
        return comb(self.n_groups, self.n_test_groups)


class WalkForwardCV:
    """Walk-forward cross-validation with fixed IS/OOS windows.

    Generates rolling splits where each in-sample (IS) window is followed
    by an out-of-sample (OOS) window. The windows step forward by `step`
    samples each iteration.

    Args:
        is_window: In-sample window size (number of samples).
        oos_window: Out-of-sample window size (number of samples).
        step: Step size between consecutive splits. Defaults to oos_window.
    """

    def __init__(
        self, is_window: int, oos_window: int, step: int | None = None
    ) -> None:
        if is_window < 1:
            raise ValueError(f"is_window must be >= 1, got {is_window}")
        if oos_window < 1:
            raise ValueError(f"oos_window must be >= 1, got {oos_window}")
        self.is_window = is_window
        self.oos_window = oos_window
        self.step = step or oos_window

    def split(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate rolling IS/OOS splits.

        Args:
            X: Array of shape (n_samples, ...). Only length is used.
            y: Ignored. Present for API compatibility.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(X)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        start = 0
        while start + self.is_window + self.oos_window <= n:
            is_start = start
            is_end = start + self.is_window
            oos_start = is_end
            oos_end = min(is_end + self.oos_window, n)

            train_idx = np.arange(is_start, is_end)
            test_idx = np.arange(oos_start, oos_end)
            splits.append((train_idx, test_idx))

            start += self.step

        return splits
