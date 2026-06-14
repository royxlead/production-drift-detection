"""ADWIN-Style Adaptive Windowing drift detector.

Implements a simplified but statistically credible approximation of the
ADWIN (Adaptive Windowing) algorithm for streaming drift detection.

The algorithm maintains a window of recent observations and adaptively
shrinks it when a statistically significant change in the mean is detected.
"""

from typing import List, Optional, Tuple

import numpy as np

from production_drift_detection.detectors.base import BaseDetector
from production_drift_detection.utils.validation import validate_array


class ADWINDetector(BaseDetector):
    """ADWIN-Style Adaptive Windowing drift detector for streaming data.

    Maintains an adaptive window of recent values and detects changes
    in distribution by comparing the means of two sub-windows.

    This is a simplified but statistically credible implementation based on
    the ADWIN algorithm (Bifet & Gavalda, 2007).

    Parameters
    ----------
    threshold : float, optional
        Alert threshold for drift score, by default 0.1.
    name : str, optional
        Detector name.
    delta : float, optional
        Confidence parameter for the change detection test.
        Lower values make detection more conservative, by default 0.05.
    max_window_size : int, optional
        Maximum window size, by default 1000.
    min_window_size : int, optional
        Minimum window size, by default 10.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        name: Optional[str] = None,
        delta: float = 0.05,
        max_window_size: int = 1000,
        min_window_size: int = 10,
    ):
        super().__init__(threshold=threshold, name=name or "ADWIN")
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._window: List[float] = []
        self._total: float = 0.0
        self._detected_change_points: List[int] = []
        self._n_detections: int = 0

    @property
    def window_size(self) -> int:
        """Current number of elements in the adaptive window."""
        return len(self._window)

    def fit(self, reference_data):
        """Fit ADWIN by initializing the window with reference data.

        Parameters
        ----------
        reference_data : array-like
            Reference data to initialize the window.

        Returns
        -------
        ADWINDetector
            Self for method chaining.
        """
        data = validate_array(reference_data, name="reference_data")
        if data.ndim > 1:
            # For multivariate data, use the mean across features
            data = np.mean(data, axis=1)

        # Initialize window with reference data
        self._window = data.tolist()
        self._total = float(np.sum(data))

        # Trim window if too large
        if len(self._window) > self.max_window_size:
            excess = len(self._window) - self.max_window_size
            removed = self._window[:excess]
            self._window = self._window[excess:]
            self._total -= sum(removed)

        self._reference_data = data
        self._fitted = True
        self._logger.info(
            f"Fitted ADWIN detector with window size {len(self._window)}"
        )
        return self

    def _compute_score(self, reference: np.ndarray, batch: np.ndarray) -> float:
        """ADWIN computes score by checking if adding batch causes detection.

        For ADWIN, the score is the maximum cut magnitude found during
        change detection.
        """
        # ADWIN doesn't use reference vs batch in the traditional sense
        # Instead, it maintains an adaptive window. We simulate a score
        # by measuring the effect of adding this batch data.
        if batch.ndim > 1:
            batch_flat = np.mean(batch, axis=1)
        else:
            batch_flat = batch

        # Simulate ADWIN: compute how much means would shift
        if not self._window:
            return 0.0

        # Calculate the current window statistics
        window_arr = np.array(self._window[-min(len(self._window), 100):])
        batch_arr = np.array(batch_flat)

        # The score is the absolute normalized difference in means
        current_mean = np.mean(window_arr)
        batch_mean = np.mean(batch_arr)
        current_std = max(np.std(window_arr), 1e-10)

        normalized_diff = abs(current_mean - batch_mean) / current_std
        return float(normalized_diff)

    def update(self, batch):
        """Update ADWIN window with new batch and detect drift.

        Overrides the base update to implement adaptive windowing.

        Parameters
        ----------
        batch : array-like
            New data batch.
        """
        batch_arr = validate_array(batch, name="batch")
        if batch_arr.ndim > 1:
            batch_flat = np.mean(batch_arr, axis=1)
        else:
            batch_flat = batch_arr

        for value in batch_flat:
            self._add_element(value)

    def _add_element(self, value: float) -> None:
        """Add a single element and check for drift."""
        self._window.append(value)
        self._total += value

        # Check for drift by comparing sub-windows
        if len(self._window) >= self.min_window_size * 2:
            self._detect_change()

        # Trim window if too large
        if len(self._window) > self.max_window_size:
            removed = self._window.pop(0)
            self._total -= removed

    def _detect_change(self) -> bool:
        """Check for a change point by examining all split points.

        Returns
        -------
        bool
            True if change detected, False otherwise.
        """
        n = len(self._window)

        for split in range(self.min_window_size, n - self.min_window_size + 1):
            left = self._window[:split]
            right = self._window[split:]

            n1 = len(left)
            n2 = len(right)
            mu1 = np.mean(left)
            mu2 = np.mean(right)

            # Compute the threshold using the ADWIN inequality
            eps = self._compute_epsilon(n1, n2)

            if abs(mu1 - mu2) > eps:
                # Change detected — shrink window
                self._window = right
                self._total = sum(right)
                self._n_detections += 1
                self._detected_change_points.append(len(self._window))

                self._logger.info(
                    f"ADWIN change detected at window size {n}, "
                    f"means: {mu1:.4f} vs {mu2:.4f}, eps={eps:.4f}"
                )
                return True

        return False

    def _compute_epsilon(self, n1: int, n2: int) -> float:
        """Compute the ADWIN change detection threshold.

        Uses the Hoeffding bound-based inequality from the ADWIN paper.

        Parameters
        ----------
        n1 : int
            Size of left sub-window.
        n2 : int
            Size of right sub-window.

        Returns
        -------
        float
            Threshold for detecting change.
        """
        n = n1 + n2
        # Reduction factor
        m = 1.0 / n1 + 1.0 / n2
        # ADWIN threshold
        eps = np.sqrt(m * np.log(2.0 / self.delta) / (2 * n))
        return eps

    def score(self, batch) -> float:
        """Compute ADWIN score.

        ADWIN's score represents the normalized mean difference between
        the current window and the new batch.

        Parameters
        ----------
        batch : array-like
            Data batch.

        Returns
        -------
        float
            Normalized drift score.
        """
        if not self._fitted or self._reference_data is None:
            raise RuntimeError("Detector must be fitted before scoring. Call fit() first.")

        batch_arr = validate_array(batch, name="batch")
        if batch_arr.ndim > 1:
            batch_flat = np.mean(batch_arr, axis=1)
        else:
            batch_flat = batch_arr

        if not self._window:
            return 0.0

        window_arr = np.array(self._window[-100:])  # Use recent 100
        batch_mean = np.mean(batch_flat)
        window_mean = np.mean(window_arr)
        window_std = max(np.std(window_arr), 1e-10)

        normalized_diff = abs(window_mean - batch_mean) / window_std
        return float(normalized_diff)

    def detect(self, batch):
        result = super().detect(batch)
        result["window_size"] = len(self._window)
        result["n_detections"] = self._n_detections
        return result

    def summary(self):
        base = super().summary()
        base.update({
            "window_size": len(self._window),
            "max_window_size": self.max_window_size,
            "n_detections": self._n_detections,
            "delta": self.delta,
            "current_mean": float(np.mean(self._window)) if self._window else None,
        })
        return base

    def reset(self):
        super().reset()
        self._window = []
        self._total = 0.0
        self._detected_change_points = []
        self._n_detections = 0
