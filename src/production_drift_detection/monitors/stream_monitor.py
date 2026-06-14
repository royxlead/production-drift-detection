"""Stream monitor for batch ingestion, drift tracking, and alerting.

The ``StreamMonitor`` orchestrates drift detection across streaming data
by coordinating multiple detectors, tracking results over time, and
triggering alerts.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from production_drift_detection.alerts.rules import AlertEngine, BaseRule, ThresholdRule
from production_drift_detection.alerts.schemas import Alert, Severity
from production_drift_detection.detectors.base import BaseDetector
from production_drift_detection.detectors.kl import KLDivergenceDetector
from production_drift_detection.detectors.mmd import MMDDetector
from production_drift_detection.detectors.psi import PSIDetector
from production_drift_detection.data.synthetic_drift import DriftGenerator
from production_drift_detection.utils.logging import get_logger
from production_drift_detection.utils.validation import validate_array


class StreamMonitor:
    """Orchestrate drift detection across streaming data.

    Manages multiple drift detectors, ingests batches sequentially,
    tracks drift history, and triggers alerts when drift is detected.

    Parameters
    ----------
    detectors : dict of str to BaseDetector, optional
        Detectors to use, by default creates KL, PSI, and MMD.
    alert_engine : AlertEngine, optional
        Alert evaluation engine.
    window_size : int, optional
        Rolling window for statistics, by default 10.
    name : str, optional
        Monitor name.
    """

    def __init__(
        self,
        detectors: Optional[Dict[str, BaseDetector]] = None,
        alert_engine: Optional[AlertEngine] = None,
        window_size: int = 10,
        name: Optional[str] = None,
    ):
        self.detectors = detectors or {
            "kl": KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
            "psi": PSIDetector(threshold=0.1, n_bins=10),
            "mmd": MMDDetector(threshold=0.05),
        }
        self.alert_engine = alert_engine or AlertEngine(rules=[ThresholdRule()])
        self.window_size = window_size
        self.name = name or "StreamMonitor"

        self._history: Dict[str, List[float]] = {
            name: [] for name in self.detectors
        }
        self._batch_timestamps: List[pd.Timestamp] = []
        self._alerts: List[Alert] = []
        self._batch_count: int = 0
        self._fitted: bool = False

        self._logger = get_logger(f"production_drift_detection.{self.name}")

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame]) -> "StreamMonitor":
        """Fit all detectors on reference data.

        Parameters
        ----------
        reference_data : array-like
            Reference (training) distribution data.

        Returns
        -------
        StreamMonitor
            Self for method chaining.
        """
        data = validate_array(reference_data, name="reference_data")

        for name, detector in self.detectors.items():
            detector.fit(data)

        self._fitted = True
        self._logger.info(
            f"Fitted {len(self.detectors)} detectors on reference data "
            f"with shape {data.shape}"
        )
        return self

    def process_batch(self, batch: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Process a single data batch through all detectors.

        Parameters
        ----------
        batch : array-like
            Current data batch.

        Returns
        -------
        dict
            Results with scores, alerts, and status per detector.
        """
        batch = validate_array(batch, name="batch")
        if not self._fitted:
            if self._batch_count == 0:
                self.fit(batch)
                return {"status": "initialized", "scores": {}, "alerts": [], "batch": self._batch_count}
            else:
                raise RuntimeError("Monitor must be fitted before processing batches.")

        self._batch_count += 1
        self._batch_timestamps.append(pd.Timestamp.now())

        results = {"scores": {}, "alerts": [], "drift_detected": False}

        for det_name, detector in self.detectors.items():
            # Detect drift
            detection = detector.detect(batch)
            results["scores"][det_name] = detection["score"]

            # Update history
            self._history[det_name].append(detection["score"])

            # Evaluate alert rules
            alerts = self.alert_engine.evaluate(
                detector_name=det_name,
                score=detection["score"],
                threshold=detector.threshold,
                batch_index=self._batch_count,
            )

            for alert in alerts:
                results["alerts"].append(alert.to_dict())
                self._alerts.append(alert)
                results["drift_detected"] = True

            # Also add detector's own detection if triggered
            if detection["drift_detected"]:
                results["drift_detected"] = True

        results["batch"] = self._batch_count
        results["status"] = self._get_overall_status(results["scores"])

        return results

    def _get_overall_status(self, scores: Dict[str, float]) -> str:
        """Determine overall monitoring status from detector scores.

        Parameters
        ----------
        scores : dict
            Current scores from all detectors.

        Returns
        -------
        str
            Severity level.
        """
        severities = []
        for name, detector in self.detectors.items():
            score = scores.get(name, 0.0)
            severities.append(detector._classify_severity(score))

        if any(s == "critical" for s in severities):
            return "critical"
        elif any(s == "warning" for s in severities):
            return "warning"
        elif any(s == "watch" for s in severities):
            return "watch"
        return "healthy"

    def get_history(self, detector_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Get drift score history.

        Parameters
        ----------
        detector_name : str, optional
            If provided, return only this detector's history.

        Returns
        -------
        dict of str to list of float
            Detector score histories.
        """
        if detector_name:
            return {detector_name: self._history.get(detector_name, [])}
        return dict(self._history)

    def get_alerts(
        self,
        severity: Optional[Severity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get triggered alerts.

        Parameters
        ----------
        severity : Severity, optional
            Filter by severity.
        limit : int, optional
            Maximum alerts to return.

        Returns
        -------
        list of Alert
            Alert objects.
        """
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def summary(self) -> Dict[str, Any]:
        """Return a comprehensive summary of monitoring state.

        Returns
        -------
        dict
            Summary with detector summaries and alert statistics.
        """
        return {
            "name": self.name,
            "batch_count": self._batch_count,
            "fitted": self._fitted,
            "detectors": {name: det.summary() for name, det in self.detectors.items()},
            "total_alerts": len(self._alerts),
            "alerts_by_severity": {
                level: len([a for a in self._alerts if a.severity.value == level])
                for level in ["healthy", "watch", "warning", "critical"]
            },
            "current_status": "healthy",
        }

    def export_results(self, format: str = "json", filepath: Optional[str] = None) -> Union[str, pd.DataFrame, None]:
        """Export monitoring results to JSON, CSV, or both.

        Parameters
        ----------
        format : str, optional
            "json" or "csv", by default "json".
        filepath : str, optional
            If provided, writes the export to the given file path.

        Returns
        -------
        str, pd.DataFrame, or None
            Exported results. Returns None when filepath is provided.
        """
        data = []
        for batch_idx in range(len(self._batch_timestamps)):
            row = {"batch": batch_idx + 1, "timestamp": self._batch_timestamps[batch_idx]}
            for det_name in self.detectors:
                row[f"{det_name}_score"] = self._history[det_name][batch_idx] if batch_idx < len(self._history[det_name]) else None
            data.append(row)

        df = pd.DataFrame(data)

        if filepath:
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient="records", date_format="iso", indent=2)
            self._logger.info(f"Results exported to {filepath}")
            return None

        if format == "csv":
            return df
        else:
            return df.to_json(orient="records", date_format="iso")

    def reset(self) -> None:
        """Reset the monitor to initial state."""
        self._history = {name: [] for name in self.detectors}
        self._batch_timestamps = []
        self._alerts = []
        self._batch_count = 0
        self._fitted = False

        for detector in self.detectors.values():
            detector.reset()

        self._logger.info("StreamMonitor reset")
