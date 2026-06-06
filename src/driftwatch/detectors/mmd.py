"""Maximum Mean Discrepancy (MMD) drift detector for multivariate distributions."""

from typing import Optional

import numpy as np

from driftwatch.detectors.base import BaseDetector
from driftwatch.utils.stats import compute_rbf_kernel
from driftwatch.utils.validation import validate_array


class MMDDetector(BaseDetector):
    """Drift detector based on Maximum Mean Discrepancy (MMD).

    MMD uses a kernel two-sample test to detect whether two distributions
    are different. Supports multivariate data with RBF kernel.

    Parameters
    ----------
    threshold : float, optional
        Alert threshold, by default 0.05.
    name : str, optional
        Detector name.
    bandwidth : float, optional
        RBF kernel bandwidth. If None, uses median heuristic.
    subsample : int, optional
        Subsample size for computing MMD (for efficiency). If None, uses all data.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        name: Optional[str] = None,
        bandwidth: Optional[float] = None,
        subsample: Optional[int] = None,
    ):
        super().__init__(threshold=threshold, name=name or "MMD")
        self.bandwidth = bandwidth
        self.subsample = subsample
        self._reference_kernel_xx: Optional[float] = None
        self._reference_gram: Optional[np.ndarray] = None

    def fit(self, reference_data):
        data = validate_array(reference_data, name="reference_data")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Handle large reference sets
        if self.subsample is not None and len(data) > self.subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(data), self.subsample, replace=False)
            data = data[idx]

        self._reference_data = data

        # Precompute reference kernel terms for efficiency
        K_xx = compute_rbf_kernel(data, data, bandwidth=self.bandwidth)
        n = len(data)
        # Sum of K_xx excluding diagonal
        self._reference_kernel_xx = (np.sum(K_xx) - n) / (n * (n - 1))
        self._reference_gram = K_xx

        self._fitted = True
        self._logger.info(
            f"Fitted MMD detector with {n} reference samples, "
            f"bandwidth={'auto' if self.bandwidth is None else f'{self.bandwidth:.4f}'}"
        )
        return self

    def _compute_score(self, reference: np.ndarray, batch: np.ndarray) -> float:
        """Compute MMD^2 (biased estimator)."""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)

        # Subsample batch if needed
        if self.subsample is not None and len(batch) > self.subsample:
            rng = np.random.default_rng()
            idx = rng.choice(len(batch), self.subsample, replace=False)
            batch = batch[idx]

        n = len(reference)
        m = len(batch)

        # Compute kernels
        K_xx = self._reference_gram
        K_yy = compute_rbf_kernel(batch, batch, bandwidth=self.bandwidth)
        K_xy = compute_rbf_kernel(reference, batch, bandwidth=self.bandwidth)

        # MMD^2 = 1/(n^2) * sum(K_xx) - 2/(nm) * sum(K_xy) + 1/(m^2) * sum(K_yy)
        mmd2 = (
            (np.sum(K_xx) - n) / (n * (n - 1))
            - 2 * np.sum(K_xy) / (n * m)
            + (np.sum(K_yy) - m) / (m * (m - 1))
        )

        return float(max(0.0, mmd2))

    def summary(self):
        base = super().summary()
        base.update({
            "bandwidth": self.bandwidth,
            "subsample": self.subsample,
            "reference_size": len(self._reference_data) if self._reference_data is not None else 0,
        })
        return base
