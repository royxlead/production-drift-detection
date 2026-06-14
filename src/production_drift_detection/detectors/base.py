"""Base detector class defining the unified API for all drift detectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from production_drift_detection.utils.logging import get_logger
from production_drift_detection.utils.validation import validate_array


class BaseDetector(ABC):
    """Abstract base class for all drift detectors.

    All detectors in ProductionDriftDetection expose a consistent API:

    - ``fit(reference_data)`` — store reference distribution
    - ``update(batch)`` — update internal state with new batch
    - ``score(batch)`` — compute drift score without raising alerts
    - ``detect(batch)`` — compute score and check alert thresholds
    - ``summary()`` — return current state as a dictionary

    Parameters
    ----------
    threshold : float, optional
        Alert threshold for drift detection, by default 0.1.
    name : str, optional
        Detector name, by default class name.
    """

    def __init__(self, threshold: float = 0.1, name: Optional[str] = None):
        self.threshold = threshold
        self.name = name or self.__class__.__name__
        self._fitted = False
        self._reference_data: Optional[np.ndarray] = None
        self._scores: List[float] = []
        self._alerts: List[Dict[str, Any]] = []
        self._logger = get_logger(f"production_drift_detection.{self.name}")

    @abstractmethod
    def _compute_score(self, reference: np.ndarray, batch: np.ndarray) -> float:
        """Compute the raw drift score between reference and batch.

        Subclasses must implement this method.

        Parameters
        ----------
        reference : np.ndarray
            Reference distribution.
        batch : np.ndarray
            Current batch to evaluate.

        Returns
        -------
        float
            Drift score.
        """
        raise NotImplementedError

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame, list]) -> "BaseDetector":
        """Fit the detector on reference (training) data.

        Parameters
        ----------
        reference_data : array-like
            Reference distribution data.

        Returns
        -------
        BaseDetector
            Self for method chaining.
        """
        self._reference_data = validate_array(reference_data, name="reference_data")
        self._fitted = True
        self._logger.info(f"Fitted on reference data with shape {self._reference_data.shape}")
        return self

    def update(self, batch: Union[np.ndarray, pd.DataFrame, list]) -> None:
        """Update detector state with a new batch (post-fit operations only).

        Base implementation stores the score. Subclasses may override
        for stateful detectors like ADWIN. Note: does NOT append to
        ``_scores`` to avoid double-counting with ``detect()``.

        Parameters
        ----------
        batch : array-like
            New data batch.
        """
        self.score(batch)

    def score(self, batch: Union[np.ndarray, pd.DataFrame, list]) -> float:
        """Compute drift score for a batch without triggering alerts.

        Parameters
        ----------
        batch : array-like
            Data batch to evaluate.

        Returns
        -------
        float
            Drift score.
        """
        if not self._fitted or self._reference_data is None:
            raise RuntimeError("Detector must be fitted before scoring. Call fit() first.")

        batch_arr = validate_array(batch, name="batch")
        score = self._compute_score(self._reference_data, batch_arr)
        return score

    def detect(self, batch: Union[np.ndarray, pd.DataFrame, list]) -> Dict[str, Any]:
        """Compute drift score and return detection result with alert info.

        Parameters
        ----------
        batch : array-like
            Data batch to evaluate.

        Returns
        -------
        dict
            Detection result with keys: score, threshold, drift_detected, name.
        """
        score = self.score(batch)
        drift_detected = score > self.threshold
        result = {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "drift_detected": drift_detected,
            "severity": self._classify_severity(score),
        }
        self._scores.append(score)
        if drift_detected:
            alert = {
                **result,
                "timestamp": pd.Timestamp.now(),
            }
            self._alerts.append(alert)
            self._logger.warning(
                f"Drift detected! Score={score:.4f}, Threshold={self.threshold:.4f}"
            )
        return result

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the detector's current state.

        Returns
        -------
        dict
            Summary with key metrics.
        """
        return {
            "name": self.name,
            "fitted": self._fitted,
            "threshold": self.threshold,
            "num_scores": len(self._scores),
            "num_alerts": len(self._alerts),
            "mean_score": float(np.mean(self._scores)) if self._scores else None,
            "max_score": float(np.max(self._scores)) if self._scores else None,
            "current_status": self._classify_severity(
                float(np.mean(self._scores[-5:])) if len(self._scores) >= 5 else (self._scores[-1] if self._scores else 0)
            ),
        }

    def _classify_severity(self, score: float) -> str:
        """Classify drift severity based on score relative to threshold.

        Parameters
        ----------
        score : float
            Drift score.

        Returns
        -------
        str
            Severity level: "healthy", "watch", "warning", or "critical".
        """
        ratio = score / max(self.threshold, 1e-10)
        if ratio < 0.5:
            return "healthy"
        elif ratio < 1.0:
            return "watch"
        elif ratio < 2.0:
            return "warning"
        else:
            return "critical"

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._fitted = False
        self._reference_data = None
        self._scores = []
        self._alerts = []
        self._logger.info("Detector reset")
