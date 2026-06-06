"""Evaluation metrics for drift detection systems.

Provides utilities for measuring detection latency, false positive rate,
stability, sensitivity, and confidence early-warning effectiveness.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from driftwatch.detectors.base import BaseDetector
from driftwatch.utils.logging import get_logger


class DetectionMetrics:
    """Namespace for detection evaluation metrics.

    Provides static methods for computing detection latency, false positive rate,
    stability, sensitivity, and comprehensive detector evaluation.

    All methods are also available as standalone functions for direct import.
    """

    @staticmethod
    def compute_detection_latency(
        scores: np.ndarray,
        threshold: float,
        drift_start_idx: int,
    ) -> Dict[str, Any]:
        """Compute how quickly drift is detected after it begins.

        Parameters
        ----------
        scores : np.ndarray
            Drift scores over time.
        threshold : float
            Detection threshold.
        drift_start_idx : int
            Index where drift was introduced.

        Returns
        -------
        dict
            Latency metrics.
        """
        if drift_start_idx >= len(scores):
            return {"error": "drift_start_idx exceeds score length"}

        post_drift = scores[drift_start_idx:]
        detection_indices = np.where(post_drift > threshold)[0]

        if len(detection_indices) == 0:
            return {
                "drift_detected": False,
                "detection_latency_batches": None,
                "detection_latency_samples": None,
                "drift_start_idx": drift_start_idx,
            }

        first_detection = detection_indices[0]
        return {
            "drift_detected": True,
            "detection_latency_batches": int(first_detection),
            "detection_latency_samples": int(first_detection),
            "drift_start_idx": drift_start_idx,
            "num_detections": int(len(detection_indices)),
            "detection_rate": float(len(detection_indices) / len(post_drift)),
        }

    @staticmethod
    def compute_false_positive_rate(
        scores: np.ndarray,
        threshold: float,
        drift_start_idx: int,
    ) -> Dict[str, Any]:
        """Compute false positive rate before drift introduction.

        Parameters
        ----------
        scores : np.ndarray
            Drift scores over time.
        threshold : float
            Detection threshold.
        drift_start_idx : int
            Index where drift was introduced.

        Returns
        -------
        dict
            FPR metrics.
        """
        if drift_start_idx >= len(scores):
            drift_start_idx = len(scores)

        pre_drift = scores[:drift_start_idx]
        if len(pre_drift) == 0:
            return {"false_positive_rate": 0.0, "false_positives": 0, "total_pre_drift": 0}

        false_positives = np.sum(pre_drift > threshold)
        fpr = false_positives / len(pre_drift)

        return {
            "false_positive_rate": float(fpr),
            "false_positives": int(false_positives),
            "total_pre_drift": int(len(pre_drift)),
        }

    @staticmethod
    def compute_detection_stability(
        scores: np.ndarray,
        window_size: int = 5,
    ) -> Dict[str, Any]:
        """Compute the stability of drift scores over time.

        A stable detector produces scores that don't oscillate wildly.

        Parameters
        ----------
        scores : np.ndarray
            Drift scores over time.
        window_size : int, optional
            Rolling window size, by default 5.

        Returns
        -------
        dict
            Stability metrics.
        """
        if len(scores) < window_size + 1:
            return {"stability": 1.0, "status": "insufficient_data"}

        cv = np.std(scores) / max(np.mean(scores), 1e-10)

        rolling_std = np.array([
            np.std(scores[max(0, i - window_size): i + 1])
            for i in range(len(scores))
        ])
        mean_volatility = float(np.mean(rolling_std))

        autocorr = float(np.corrcoef(scores[:-1], scores[1:])[0, 1]) if len(scores) > 1 else 0

        return {
            "stability": float(1.0 / (1.0 + cv)),
            "coefficient_of_variation": float(cv),
            "mean_volatility": mean_volatility,
            "autocorrelation": autocorr,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }

    @staticmethod
    def compute_sensitivity_to_drift(
        detector: BaseDetector,
        drift_generator: Any,
        magnitudes: Optional[List[float]] = None,
        n_trials: int = 5,
        n_samples: int = 200,
    ) -> Dict[str, Any]:
        """Measure how detector scores change with drift magnitude.

        Parameters
        ----------
        detector : BaseDetector
            Detector to evaluate.
        drift_generator : DriftGenerator
            Drift generator instance.
        magnitudes : list of float, optional
            Drift magnitudes to test.
        n_trials : int, optional
            Trials per magnitude, by default 5.
        n_samples : int, optional
            Samples per trial, by default 200.

        Returns
        -------
        dict
            Sensitivity metrics.
        """
        if magnitudes is None:
            magnitudes = [0.0, 0.5, 1.0, 2.0, 3.0]

        results: Dict[str, Any] = {"magnitudes": magnitudes, "mean_scores": [], "std_scores": []}

        for mag in magnitudes:
            scores = []
            for trial in range(n_trials):
                if mag == 0:
                    batch = drift_generator.generate_reference()[:n_samples]
                else:
                    batch = drift_generator.covariate_shift(
                        n_samples=n_samples, shift_magnitude=mag
                    )
                score = detector.score(batch)
                scores.append(score)

            results["mean_scores"].append(float(np.mean(scores)))
            results["std_scores"].append(float(np.std(scores)))

        if len(magnitudes) > 1 and results["mean_scores"][-1] > results["mean_scores"][0]:
            sensitivity = (
                (results["mean_scores"][-1] - results["mean_scores"][0])
                / max(magnitudes[-1] - magnitudes[0], 1e-10)
            )
        else:
            sensitivity = 0.0

        results["sensitivity"] = float(sensitivity)
        return results

    @staticmethod
    def evaluate_detector(
        detector: BaseDetector,
        reference_data: np.ndarray,
        clean_batches: List[np.ndarray],
        drifted_batches: List[np.ndarray],
        drift_start_batch: int,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a detector.

        Parameters
        ----------
        detector : BaseDetector
            Detector to evaluate.
        reference_data : np.ndarray
            Reference distribution.
        clean_batches : list of np.ndarray
            Batches without drift.
        drifted_batches : list of np.ndarray
            Batches with drift.
        drift_start_batch : int
            Batch index where drift starts.

        Returns
        -------
        dict
            Evaluation results.
        """
        detector.fit(reference_data)

        all_scores = []
        for batch in clean_batches:
            all_scores.append(detector.score(batch))
        for batch in drifted_batches:
            all_scores.append(detector.score(batch))

        scores_arr = np.array(all_scores)

        fpr = DetectionMetrics.compute_false_positive_rate(scores_arr, detector.threshold, drift_start_batch)
        latency = DetectionMetrics.compute_detection_latency(scores_arr, detector.threshold, drift_start_batch)
        stability = DetectionMetrics.compute_detection_stability(scores_arr)

        return {
            "detector": detector.name,
            "threshold": detector.threshold,
            "false_positive_rate": fpr,
            "detection_latency": latency,
            "stability": stability,
            "all_scores": all_scores,
        }


# Module-level aliases for backward compatibility
# These allow: from driftwatch.evaluation.metrics import compute_detection_latency
compute_detection_latency = DetectionMetrics.compute_detection_latency
compute_false_positive_rate = DetectionMetrics.compute_false_positive_rate
compute_detection_stability = DetectionMetrics.compute_detection_stability
compute_sensitivity_to_drift = DetectionMetrics.compute_sensitivity_to_drift
evaluate_detector = DetectionMetrics.evaluate_detector


__all__ = [
    "DetectionMetrics",
    "compute_detection_latency",
    "compute_false_positive_rate",
    "compute_detection_stability",
    "compute_sensitivity_to_drift",
    "evaluate_detector",
]
