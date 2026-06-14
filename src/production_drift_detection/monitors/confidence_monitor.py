"""Confidence monitor for tracking model confidence, entropy, and margin over time.

The ``ConfidenceMonitor`` monitors prediction confidence trends and detects
early degradation signals before accuracy visibly drops.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from production_drift_detection.utils.logging import get_logger
from production_drift_detection.utils.stats import compute_confidence, compute_entropy, compute_margin
from production_drift_detection.utils.validation import validate_array, validate_probabilities


class ConfidenceMonitor:
    """Monitor prediction confidence, entropy, and margin over time.

    Tracks how model confidence evolves across batches and detects patterns
    that may indicate distribution shift before performance degrades.

    Parameters
    ----------
    window_size : int, optional
        Rolling window for statistics, by default 10.
    name : str, optional
        Monitor name.
    confidence_threshold : float, optional
        Threshold below which confidence is considered low, by default 0.5.
    entropy_threshold : float, optional
        Threshold above which entropy indicates uncertainty, by default 0.5.
    """

    def __init__(
        self,
        window_size: int = 10,
        name: Optional[str] = None,
        confidence_threshold: float = 0.5,
        entropy_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.name = name or "ConfidenceMonitor"
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold

        self._confidence_history: List[float] = []
        self._entropy_history: List[float] = []
        self._margin_history: List[float] = []
        self._accuracy_history: List[float] = []
        self._prob_history: List[np.ndarray] = []
        self._batch_sizes: List[int] = []
        self._batch_timestamps: List[pd.Timestamp] = []

        self._logger = get_logger(f"production_drift_detection.{self.name}")

    def update(
        self,
        probabilities: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Update monitor with a batch of prediction probabilities.

        Parameters
        ----------
        probabilities : np.ndarray
            Model prediction probabilities of shape (n_samples, n_classes).
        ground_truth : np.ndarray, optional
            Ground truth labels for accuracy tracking, if available.

        Returns
        -------
        dict
            Current batch statistics.
        """
        probs = validate_probabilities(probabilities, name="probabilities")

        # Compute metrics
        confidence = compute_confidence(probs)
        entropy = compute_entropy(probs)
        margin = compute_margin(probs)

        # Store batch statistics
        batch_stats = {
            "mean_confidence": float(np.mean(confidence)),
            "std_confidence": float(np.std(confidence)),
            "mean_entropy": float(np.mean(entropy)),
            "std_entropy": float(np.std(entropy)),
            "mean_margin": float(np.mean(margin)),
            "std_margin": float(np.std(margin)),
            "n_samples": len(probs),
        }

        # Track accuracy if ground truth is available
        if ground_truth is not None:
            predictions = np.argmax(probs, axis=1)
            accuracy = float(np.mean(predictions == ground_truth))
            batch_stats["accuracy"] = accuracy
            self._accuracy_history.append(accuracy)

        # Update history
        self._confidence_history.append(float(np.mean(confidence)))
        self._entropy_history.append(float(np.mean(entropy)))
        self._margin_history.append(float(np.mean(margin)))
        self._prob_history.append(probs)
        self._batch_sizes.append(len(probs))
        self._batch_timestamps.append(pd.Timestamp.now())

        return batch_stats

    def get_trends(self) -> Dict[str, Any]:
        """Analyze confidence trends over time.

        Returns
        -------
        dict
            Trend analysis with direction, magnitude, and alerts.
        """
        if len(self._confidence_history) < 2:
            return {"status": "insufficient_data", "batches_observed": len(self._confidence_history)}

        conf = np.array(self._confidence_history)
        entropy = np.array(self._entropy_history)
        margin = np.array(self._margin_history)

        # Linear trend via first and last window
        def _trend_direction(series: np.ndarray) -> str:
            if len(series) < 2:
                return "stable"
            first_half = np.mean(series[: len(series) // 2])
            second_half = np.mean(series[len(series) // 2:])
            diff = second_half - first_half
            if abs(diff) < 0.02:
                return "stable"
            return "increasing" if diff > 0 else "decreasing"

        def _trend_magnitude(series: np.ndarray) -> float:
            half = len(series) // 2
            return float(np.mean(series[half:]) - np.mean(series[:half]))

        return {
            "confidence_trend": _trend_direction(conf),
            "confidence_magnitude": _trend_magnitude(conf),
            "entropy_trend": _trend_direction(entropy),
            "entropy_magnitude": _trend_magnitude(entropy),
            "margin_trend": _trend_direction(margin),
            "margin_magnitude": _trend_magnitude(margin),
            "current_confidence": float(conf[-1]),
            "current_entropy": float(entropy[-1]),
            "current_margin": float(margin[-1]),
            "baseline_confidence": float(conf[0]),
            "baseline_entropy": float(entropy[0]),
            "confidence_change_pct": float((conf[-1] - conf[0]) / max(conf[0], 1e-10) * 100),
            "batches_observed": len(conf),
        }

    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """Get summary of uncertainty metrics for the most recent batch.

        Includes calibration-like summaries based on prediction confidence.

        Returns
        -------
        dict
            Uncertainty metrics.
        """
        if not self._prob_history:
            return {}

        probs = self._prob_history[-1]
        confidence = compute_confidence(probs)

        # Proportion of low-confidence predictions
        low_conf_ratio = float(np.mean(confidence < self.confidence_threshold))
        high_entropy_ratio = float(
            np.mean(compute_entropy(probs) > self.entropy_threshold)
        )

        return {
            "low_confidence_ratio": low_conf_ratio,
            "high_entropy_ratio": high_entropy_ratio,
            "mean_confidence": float(np.mean(confidence)),
            "median_confidence": float(np.median(confidence)),
            "confidence_volatility": float(np.std(confidence)),
            "confidence_percentile_5": float(np.percentile(confidence, 5)),
            "confidence_percentile_95": float(np.percentile(confidence, 95)),
        }

    def degradation_detected(self) -> Tuple[bool, str]:
        """Check if confidence degradation is detected.

        Returns
        -------
        Tuple[bool, str]
            (degraded, reason) tuple.
        """
        if len(self._confidence_history) < 3:
            return False, "insufficient_data"

        # Check for significant confidence drop
        recent = np.mean(self._confidence_history[-3:])
        baseline = np.mean(self._confidence_history[:3])
        drop_pct = (baseline - recent) / max(baseline, 1e-10)

        if drop_pct > 0.15:
            return True, f"Confidence dropped {drop_pct * 100:.1f}% from baseline"

        if len(self._entropy_history) >= 3:
            recent_entropy = np.mean(self._entropy_history[-3:])
            baseline_entropy = np.mean(self._entropy_history[:3])
            if recent_entropy > baseline_entropy * 1.3:
                return True, f"Entropy increased {((recent_entropy / max(baseline_entropy, 1e-10)) - 1) * 100:.1f}% from baseline"

        return False, "stable"

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get a summary of model calibration characteristics.

        Uses confidence histograms to estimate calibration quality
        (without requiring true labels).

        Returns
        -------
        dict
            Calibration summary.
        """
        if not self._prob_history:
            return {}

        all_probs = np.vstack(self._prob_history)
        confidence = compute_confidence(all_probs)

        # Confidence distribution
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(confidence, bins=bins)

        return {
            "confidence_distribution": hist.tolist(),
            "mean_confidence_overall": float(np.mean(confidence)),
            "overconfident_ratio": float(np.mean(confidence > 0.9)),
            "underconfident_ratio": float(np.mean(confidence < 0.5)),
            "ece_approximation": float(np.std(confidence)),  # Approximate calibration error
        }

    def summary(self) -> Dict[str, Any]:
        """Return comprehensive summary of confidence monitoring.

        Returns
        -------
        dict
            Summary statistics.
        """
        trends = self.get_trends()
        uncertainty = self.get_uncertainty_metrics()
        degraded, reason = self.degradation_detected()

        return {
            "name": self.name,
            "batches_monitored": len(self._confidence_history),
            "total_samples": sum(self._batch_sizes),
            "degradation_detected": degraded,
            "degradation_reason": reason,
            "trends": trends,
            "uncertainty": uncertainty,
            "mean_accuracy": float(np.mean(self._accuracy_history)) if self._accuracy_history else None,
            "accuracy_trend": (
                "decreasing" if self._accuracy_history and len(self._accuracy_history) >= 2
                and self._accuracy_history[-1] < self._accuracy_history[0] - 0.05
                else "stable"
            ) if self._accuracy_history else None,
        }

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self._confidence_history = []
        self._entropy_history = []
        self._margin_history = []
        self._accuracy_history = []
        self._prob_history = []
        self._batch_sizes = []
        self._batch_timestamps = []
        self._logger.info("ConfidenceMonitor reset")
