"""Population Stability Index (PSI) drift detector for numerical and categorical features."""

from typing import Dict, List, Optional, Union

import numpy as np

from driftwatch.detectors.base import BaseDetector
from driftwatch.utils.stats import compute_psi_bins, smooth_distribution
from driftwatch.utils.validation import validate_array


class PSIDetector(BaseDetector):
    """Drift detector based on Population Stability Index (PSI).

    PSI measures the stability of a feature distribution over time by
    comparing bin proportions. Supports both numerical and categorical
    features with per-feature and aggregate drift reporting.

    Parameters
    ----------
    threshold : float, optional
        Alert threshold, by default 0.1. Standard convention:
        PSI < 0.1 — no significant change
        0.1 <= PSI < 0.25 — moderate change
        PSI >= 0.25 — significant change
    name : str, optional
        Detector name.
    n_bins : int, optional
        Number of bins for numerical features, by default 10.
    bin_strategy : str, optional
        Binning strategy ("quantile" or "uniform"), by default "quantile".
    feature_names : list of str, optional
        Names of features for reporting.
    per_feature : bool, optional
        Whether to compute per-feature PSI, by default True.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        name: Optional[str] = None,
        n_bins: int = 10,
        bin_strategy: str = "quantile",
        feature_names: Optional[List[str]] = None,
        per_feature: bool = True,
    ):
        super().__init__(threshold=threshold, name=name or "PSI")
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.feature_names = feature_names
        self.per_feature = per_feature
        self._reference_props: Optional[Dict[int, tuple]] = None
        self._n_features: int = 0
        self._per_feature_scores: List[Dict[str, float]] = []

    def fit(self, reference_data):
        data = validate_array(reference_data, name="reference_data")

        # Handle 1D arrays
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._n_features = data.shape[1]

        # Compute reference bin proportions for each feature
        self._reference_props = {}
        for col in range(self._n_features):
            feature_data = data[:, col]
            # Use quantiles of this feature for bin edges
            bin_edges = np.percentile(
                feature_data, np.linspace(0, 100, self.n_bins + 1)
            )
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.array([np.min(feature_data), np.max(feature_data) + 1e-6])
            counts, _ = np.histogram(feature_data, bins=bin_edges)
            props = smooth_distribution(counts.astype(float))
            self._reference_props[col] = (bin_edges, props)

        if self.feature_names is None:
            if self._n_features == 1:
                self.feature_names = ["feature_0"]
            else:
                self.feature_names = [f"feature_{i}" for i in range(self._n_features)]

        self._reference_data = data
        self._fitted = True
        self._logger.info(
            f"Fitted PSI detector with {self._n_features} features, {self.n_bins} bins each"
        )
        return self

    def _compute_psi_single(self, ref_props: np.ndarray, actual_props: np.ndarray) -> float:
        """Compute PSI for a single feature."""
        # PSI = sum((actual - ref) * ln(actual / ref))
        p = ref_props
        q = actual_props
        psi = np.sum((q - p) * np.log(q / p))
        return float(max(0.0, psi))

    def _compute_score(self, reference: np.ndarray, batch: np.ndarray) -> float:
        """Compute aggregate PSI across all features."""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)

        n_features = min(batch.shape[1], self._n_features)
        feature_scores = {}

        total_psi = 0.0
        for col in range(n_features):
            bin_edges, ref_props = self._reference_props[col]
            batch_feature = batch[:, col]
            batch_counts, _ = np.histogram(batch_feature, bins=bin_edges)
            batch_props = smooth_distribution(batch_counts.astype(float))

            psi = self._compute_psi_single(ref_props, batch_props)
            feature_scores[self.feature_names[col]] = psi
            total_psi += psi

        avg_psi = total_psi / n_features

        if self.per_feature:
            self._per_feature_scores.append(feature_scores)

        return avg_psi

    def get_feature_scores(self) -> List[Dict[str, float]]:
        """Return per-feature PSI scores for all scored batches.

        Returns
        -------
        list of dict
            Per-feature PSI scores for each batch.
        """
        return self._per_feature_scores

    def summary(self):
        base = super().summary()
        base.update({
            "n_features": self._n_features,
            "n_bins": self.n_bins,
            "bin_strategy": self.bin_strategy,
            "per_feature": self.per_feature,
        })
        if self._per_feature_scores:
            latest = self._per_feature_scores[-1]
            sorted_features = sorted(latest.items(), key=lambda x: x[1], reverse=True)
            base["top_drifted_features"] = sorted_features[:5]
        return base
