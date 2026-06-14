"""KL Divergence drift detector for categorical and probability distributions."""

from typing import Optional

import numpy as np

from production_drift_detection.detectors.base import BaseDetector
from production_drift_detection.utils.stats import smooth_distribution
from production_drift_detection.utils.validation import validate_array


class KLDivergenceDetector(BaseDetector):
    """Drift detector based on Kullback-Leibler Divergence.

    Measures the KL divergence between reference and actual probability
    distributions. Best suited for categorical features and probability
    outputs.

    Parameters
    ----------
    threshold : float, optional
        Alert threshold for KL divergence, by default 0.1.
    name : str, optional
        Detector name.
    smoothing : float, optional
        Laplace smoothing epsilon, by default 1e-10.
    is_categorical : bool, optional
        Whether data is categorical (histogram-based), by default True.
    n_bins : int, optional
        Number of bins for numerical data, by default 20.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        name: Optional[str] = None,
        smoothing: float = 1e-10,
        is_categorical: bool = True,
        n_bins: int = 20,
    ):
        super().__init__(threshold=threshold, name=name or "KLDivergence")
        self.smoothing = smoothing
        self.is_categorical = is_categorical
        self.n_bins = n_bins
        self._reference_pmf: Optional[np.ndarray] = None
        self._ref_hist_edges: Optional[np.ndarray] = None

    def fit(self, reference_data):
        data = validate_array(reference_data, name="reference_data")

        if self.is_categorical:
            # For categorical data, compute PMF from unique values
            values, counts = np.unique(data, return_counts=True)
            self._reference_pmf = smooth_distribution(
                counts.astype(float), eps=self.smoothing
            )
            self._ref_categories = values
        else:
            # For continuous data, bin and compute histogram
            if data.ndim > 1:
                data_flat = data.flatten()
            else:
                data_flat = data
            self._ref_hist_edges = np.percentile(
                data_flat, np.linspace(0, 100, self.n_bins + 1)
            )
            self._ref_hist_edges = np.unique(self._ref_hist_edges)
            ref_counts, _ = np.histogram(data_flat, bins=self._ref_hist_edges)
            self._reference_pmf = smooth_distribution(
                ref_counts.astype(float), eps=self.smoothing
            )
            self._ref_categories = None

        self._reference_data = data
        self._fitted = True
        self._logger.info(
            f"Fitted KL detector with {len(self._reference_pmf)} categories/bins"
        )
        return self

    def _compute_score(self, reference: np.ndarray, batch: np.ndarray) -> float:
        """Compute KL divergence safely with numerical stability."""
        # Compute batch distribution
        if self.is_categorical:
            # Match batch categories to reference categories
            batch_counts = np.zeros(len(self._ref_categories), dtype=float)
            for i, cat in enumerate(self._ref_categories):
                batch_counts[i] = np.sum(batch == cat)
            batch_pmf = smooth_distribution(batch_counts, eps=self.smoothing)
        else:
            if batch.ndim > 1:
                batch_flat = batch.flatten()
            else:
                batch_flat = batch
            batch_counts, _ = np.histogram(batch_flat, bins=self._ref_hist_edges)
            batch_pmf = smooth_distribution(batch_counts.astype(float), eps=self.smoothing)

        # Compute KL divergence: sum(P * log(P / Q))
        # where P = reference, Q = batch (or vice versa)
        # Using reference as P for consistency
        p = self._reference_pmf
        q = batch_pmf

        # Ensure numerical stability
        ratio = p / q
        kl_div = float(np.sum(p * np.log(ratio)))

        # KL divergence is always non-negative
        return max(0.0, kl_div)

    def summary(self):
        base = super().summary()
        base.update({
            "smoothing": self.smoothing,
            "is_categorical": self.is_categorical,
            "n_categories": len(self._reference_pmf) if self._reference_pmf is not None else 0,
        })
        return base
