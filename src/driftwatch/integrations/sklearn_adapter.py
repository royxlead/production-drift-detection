"""Scikit-learn integration adapter for DriftWatch.

Provides utilities for wrapping scikit-learn models with drift monitoring.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator

from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.monitors.stream_monitor import StreamMonitor


class SklearnAdapter:
    """Adapter for monitoring scikit-learn models with DriftWatch.

    Parameters
    ----------
    model : BaseEstimator
        Fitted scikit-learn model.
    stream_monitor : StreamMonitor, optional
        Monitor for data drift.
    confidence_monitor : ConfidenceMonitor, optional
        Monitor for confidence drift.
    predict_proba : bool, optional
        Whether the model supports predict_proba, by default True.
    """

    def __init__(
        self,
        model: BaseEstimator,
        stream_monitor: Optional[StreamMonitor] = None,
        confidence_monitor: Optional[ConfidenceMonitor] = None,
        predict_proba: bool = True,
    ):
        self.model = model
        self.stream_monitor = stream_monitor or StreamMonitor()
        self.confidence_monitor = confidence_monitor or ConfidenceMonitor()
        self._predict_proba = predict_proba

    def predict(self, X: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Make predictions and monitor for drift.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        ground_truth : np.ndarray, optional
            Ground truth labels, if available.

        Returns
        -------
        dict
            Predictions and monitoring results.
        """
        # Monitor data drift
        stream_result = self.stream_monitor.process_batch(X)

        # Get predictions
        if self._predict_proba and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = self.model.predict(X)
            # Create pseudo-probabilities
            n_classes = len(np.unique(predictions))
            probabilities = np.zeros((len(X), n_classes))
            probabilities[np.arange(len(X)), predictions] = 1.0

        # Monitor confidence
        conf_result = self.confidence_monitor.update(probabilities, ground_truth)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "drift": stream_result,
            "confidence": conf_result,
        }

    def summary(self) -> Dict[str, Any]:
        """Get combined summary.

        Returns
        -------
        dict
            Summary from all monitors.
        """
        return {
            "model": type(self.model).__name__,
            "stream_monitor": self.stream_monitor.summary(),
            "confidence_monitor": self.confidence_monitor.summary(),
        }
