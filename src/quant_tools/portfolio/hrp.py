"""Hierarchical portfolio optimization: HRP, HERC, NCO."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

log = logging.getLogger("quant_tools")


class HierarchicalRiskParity:
    """De Prado's HRP: cluster assets by correlation, allocate inversely to risk.

    Steps:
    1. Correlation matrix -> distance matrix
    2. Hierarchical clustering (linkage)
    3. Quasi-diagonal reordering
    4. Recursive bisection allocation
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        linkage_method: str = "ward",
    ) -> dict[str, float]:
        """Full HRP optimization.

        Args:
            returns: DataFrame with columns = asset symbols, rows = daily returns.
            linkage_method: Hierarchical clustering linkage method
                ('ward', 'single', 'complete', 'average').

        Returns:
            Dict of {symbol: weight}, weights sum to 1.0, all non-negative.
        """
        if returns.shape[1] < 2:
            # Degenerate: single asset gets 100%
            return {returns.columns[0]: 1.0}

        corr = returns.corr().values
        cov = returns.cov().values
        symbols = list(returns.columns)

        # 1. Clustering
        link = self._tree_clustering(corr, linkage_method)

        # 2. Quasi-diagonal reordering
        sorted_idx = self._quasi_diag(link)

        # 3. Recursive bisection
        weights = self._recursive_bisection(cov, sorted_idx)

        return {
            symbols[sorted_idx[i]]: round(float(weights[i]), 6)
            for i in range(len(sorted_idx))
        }

    def _tree_clustering(self, corr: np.ndarray, method: str) -> np.ndarray:
        """Hierarchical clustering on correlation-based distance.

        Distance = sqrt(0.5 * (1 - corr)), which maps [-1, 1] -> [0, 1].
        """
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        # Convert to condensed form for scipy
        condensed = squareform(dist, checks=False)
        return linkage(condensed, method=method)

    def _quasi_diag(self, link: np.ndarray) -> list[int]:
        """Reorder assets for quasi-diagonal correlation structure.

        Uses scipy.cluster.hierarchy.leaves_list which returns the
        leaf ordering from the dendrogram.
        """
        return list(leaves_list(link).astype(int))

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_idx: list[int],
    ) -> np.ndarray:
        """Allocate weights via recursive bisection of clustered covariance.

        At each level, split the sorted index list in half.
        Allocate inversely proportional to cluster variance.
        """
        n = len(sorted_idx)
        weights = np.ones(n)
        clusters = [list(range(n))]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Cluster variance = w' * Sigma * w for equal-weighted sub-portfolio
                left_var = self._cluster_variance(cov, sorted_idx, left)
                right_var = self._cluster_variance(cov, sorted_idx, right)

                # Allocate inversely proportional to variance
                total_inv_var = 1.0 / (left_var + 1e-15) + 1.0 / (right_var + 1e-15)
                alpha_left = (1.0 / (left_var + 1e-15)) / total_inv_var
                alpha_right = 1.0 - alpha_left

                for i in left:
                    weights[i] *= alpha_left
                for i in right:
                    weights[i] *= alpha_right

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            clusters = new_clusters

        return weights

    @staticmethod
    def _cluster_variance(
        cov: np.ndarray,
        sorted_idx: list[int],
        cluster: list[int],
    ) -> float:
        """Variance of an equal-weighted sub-portfolio within a cluster."""
        idx = [sorted_idx[i] for i in cluster]
        sub_cov = cov[np.ix_(idx, idx)]
        n = len(idx)
        w = np.ones(n) / n
        return float(w @ sub_cov @ w)


class HERC:
    """Hierarchical Equal Risk Contribution.

    Similar to HRP but uses equal risk contribution within each cluster
    rather than inverse-variance allocation.
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        linkage_method: str = "ward",
        risk_measure: str = "variance",
    ) -> dict[str, float]:
        """HERC optimization.

        Args:
            returns: DataFrame with columns = asset symbols, rows = daily returns.
            linkage_method: Hierarchical clustering linkage method.
            risk_measure: Risk measure for allocation ('variance' or 'std').

        Returns:
            Dict of {symbol: weight}, weights sum to 1.0.
        """
        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0}

        corr = returns.corr().values
        cov = returns.cov().values
        symbols = list(returns.columns)
        n = len(symbols)

        # Distance and clustering
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=linkage_method)

        # Get cluster assignments (cut tree at n_clusters = sqrt(n))
        n_clusters = max(2, int(np.sqrt(n)))
        cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust")

        # Within each cluster: equal risk contribution
        cluster_weights = np.zeros(n)
        cluster_risks = {}

        for c in sorted(set(cluster_labels)):
            members = [i for i in range(n) if cluster_labels[i] == c]
            if len(members) == 1:
                cluster_weights[members[0]] = 1.0
                sub_cov = cov[members[0], members[0]]
                if risk_measure == "std":
                    cluster_risks[c] = float(np.sqrt(sub_cov))
                else:
                    cluster_risks[c] = float(sub_cov)
            else:
                # Equal risk contribution within cluster
                sub_cov_mat = cov[np.ix_(members, members)]
                erc_weights = self._equal_risk_contribution(sub_cov_mat)
                for i, m in enumerate(members):
                    cluster_weights[m] = erc_weights[i]

                # Cluster-level risk
                w_sub = erc_weights
                port_var = float(w_sub @ sub_cov_mat @ w_sub)
                if risk_measure == "std":
                    cluster_risks[c] = float(np.sqrt(port_var))
                else:
                    cluster_risks[c] = port_var

        # Across clusters: inverse risk allocation
        total_inv_risk = sum(1.0 / (r + 1e-15) for r in cluster_risks.values())
        cluster_alloc = {
            c: (1.0 / (r + 1e-15)) / total_inv_risk
            for c, r in cluster_risks.items()
        }

        # Final weights
        final_weights = np.zeros(n)
        for c in sorted(set(cluster_labels)):
            members = [i for i in range(n) if cluster_labels[i] == c]
            for m in members:
                final_weights[m] = cluster_weights[m] * cluster_alloc[c]

        # Normalize to sum to 1
        total = final_weights.sum()
        if total > 0:
            final_weights /= total

        return {
            symbols[i]: round(float(final_weights[i]), 6)
            for i in range(n)
        }

    @staticmethod
    def _equal_risk_contribution(cov: np.ndarray) -> np.ndarray:
        """Compute equal risk contribution weights for a sub-portfolio.

        Uses a simplified iterative approach.
        """
        n = cov.shape[0]
        w = np.ones(n) / n

        for _ in range(100):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-15:
                break
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / (port_vol + 1e-15)
            target_rc = port_vol / n

            # Adjust weights: increase for under-contributing, decrease for over
            adjustment = target_rc / (risk_contrib + 1e-15)
            w = w * adjustment
            w = np.maximum(w, 1e-10)
            w /= w.sum()

        return w


class NCO:
    """Nested Clustered Optimization.

    Optimize within clusters first, then optimize across clusters.
    This reduces the dimensionality of the optimization problem and
    mitigates estimation error in the covariance matrix.
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        n_clusters: int | None = None,
    ) -> dict[str, float]:
        """NCO optimization.

        Args:
            returns: DataFrame with columns = asset symbols, rows = daily returns.
            n_clusters: Number of clusters. If None, uses sqrt(n_assets).

        Returns:
            Dict of {symbol: weight}, weights sum to 1.0.
        """
        symbols = list(returns.columns)
        n = len(symbols)

        if n < 2:
            return {symbols[0]: 1.0}

        cov = returns.cov().values
        mu = returns.mean().values

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(n)))
        n_clusters = min(n_clusters, n)

        # Hierarchical clustering
        corr = returns.corr().values
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="ward")
        cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust")

        # Stage 1: Optimize within each cluster (min-variance)
        intra_weights = np.zeros(n)
        cluster_returns = {}
        cluster_vols = {}

        for c in sorted(set(cluster_labels)):
            members = [i for i in range(n) if cluster_labels[i] == c]
            if len(members) == 1:
                intra_weights[members[0]] = 1.0
                cluster_returns[c] = float(mu[members[0]])
                cluster_vols[c] = float(np.sqrt(cov[members[0], members[0]]))
            else:
                sub_cov = cov[np.ix_(members, members)]
                # Min-variance within cluster
                inv_cov = np.linalg.pinv(sub_cov)
                ones = np.ones(len(members))
                denom = float(ones @ inv_cov @ ones)
                if abs(denom) < 1e-15:
                    w_sub = np.ones(len(members)) / len(members)
                else:
                    w_sub = (inv_cov @ ones) / denom
                w_sub = np.maximum(w_sub, 0)
                w_sum = w_sub.sum()
                if w_sum > 0:
                    w_sub /= w_sum

                for i, m in enumerate(members):
                    intra_weights[m] = w_sub[i]

                sub_mu = mu[members]
                cluster_returns[c] = float(w_sub @ sub_mu)
                cluster_vols[c] = float(np.sqrt(w_sub @ sub_cov @ w_sub))

        # Stage 2: Optimize across clusters (inverse variance)
        total_inv_var = sum(
            1.0 / (v**2 + 1e-15) for v in cluster_vols.values()
        )
        cluster_alloc = {
            c: (1.0 / (v**2 + 1e-15)) / total_inv_var
            for c, v in cluster_vols.items()
        }

        # Final weights
        final_weights = np.zeros(n)
        for c in sorted(set(cluster_labels)):
            members = [i for i in range(n) if cluster_labels[i] == c]
            for m in members:
                final_weights[m] = intra_weights[m] * cluster_alloc[c]

        # Normalize
        total = final_weights.sum()
        if total > 0:
            final_weights /= total

        return {
            symbols[i]: round(float(final_weights[i]), 6)
            for i in range(n)
        }


def plot_dendrogram(
    returns: pd.DataFrame,
    linkage_method: str = "ward",
):
    """Return Plotly figure of asset dendrogram.

    Args:
        returns: DataFrame with columns = asset symbols, rows = daily returns.
        linkage_method: Hierarchical clustering linkage method.

    Returns:
        plotly.graph_objects.Figure with the dendrogram.
    """
    import plotly.figure_factory as ff

    corr = returns.corr().values
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    labels = list(returns.columns)

    fig = ff.create_dendrogram(
        dist,
        labels=labels,
        linkagefun=lambda x: linkage(x, method=linkage_method),
    )
    fig.update_layout(
        title="Asset Dendrogram (Correlation-Based Distance)",
        xaxis_title="Assets",
        yaxis_title="Distance",
    )
    return fig
