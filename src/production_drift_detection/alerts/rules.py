"""Alert rules and engine for ProductionDriftDetection."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from production_drift_detection.alerts.schemas import Alert, Severity
from production_drift_detection.utils.logging import get_logger


class BaseRule(ABC):
    """Abstract base class for alert rules."""

    @abstractmethod
    def evaluate(self, detector_name: str, score: float, threshold: float, **kwargs) -> Optional[Alert]:
        """Evaluate a score against this rule.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        score : float
            Current drift score.
        threshold : float
            Detector's threshold.
        **kwargs
            Additional context.

        Returns
        -------
        Alert or None
            Alert if triggered, None otherwise.
        """
        raise NotImplementedError


class ThresholdRule(BaseRule):
    """Simple threshold-based alert rule.

    Parameters
    ----------
    warning_multiplier : float, optional
        Multiplier to determine warning threshold, by default 1.0.
    critical_multiplier : float, optional
        Multiplier to determine critical threshold, by default 2.0.
    """

    def __init__(self, warning_multiplier: float = 1.0, critical_multiplier: float = 2.0):
        self.warning_multiplier = warning_multiplier
        self.critical_multiplier = critical_multiplier

    def evaluate(self, detector_name: str, score: float, threshold: float, **kwargs) -> Optional[Alert]:
        """Evaluate score against threshold levels.

        Returns
        -------
        Alert or None
            Alert if score exceeds threshold.
        """
        warning_threshold = threshold * self.warning_multiplier
        critical_threshold = threshold * self.critical_multiplier

        if score > critical_threshold:
            return Alert(
                detector=detector_name,
                score=score,
                threshold=critical_threshold,
                severity=Severity.CRITICAL,
                explanation=f"Critical drift: score {score:.4f} exceeds critical threshold {critical_threshold:.4f}",
                metadata={"base_threshold": threshold},
            )
        elif score > warning_threshold:
            severity = Severity.WARNING if score > threshold else Severity.WATCH
            return Alert(
                detector=detector_name,
                score=score,
                threshold=warning_threshold,
                severity=severity,
                explanation=f"Score {score:.4f} exceeds threshold {warning_threshold:.4f}",
                metadata={"base_threshold": threshold},
            )
        return None


class RollingWindowRule(BaseRule):
    """Rolling window alert rule that considers recent history.

    Parameters
    ----------
    window_size : int, optional
        Size of the rolling window, by default 5.
    std_multiplier : float, optional
        Number of standard deviations for alert, by default 2.0.
    min_consecutive : int, optional
        Minimum consecutive exceedances to trigger, by default 2.
    """

    def __init__(
        self,
        window_size: int = 5,
        std_multiplier: float = 2.0,
        min_consecutive: int = 2,
    ):
        self.window_size = window_size
        self.std_multiplier = std_multiplier
        self.min_consecutive = min_consecutive
        self._history: Dict[str, List[float]] = {}

    def evaluate(self, detector_name: str, score: float, threshold: float, **kwargs) -> Optional[Alert]:
        """Evaluate score using rolling window statistics.

        Returns
        -------
        Alert or None
            Alert if anomalous pattern detected.
        """
        if detector_name not in self._history:
            self._history[detector_name] = []

        self._history[detector_name].append(score)

        # Keep only recent history
        history = self._history[detector_name]
        if len(history) > self.window_size:
            history = history[-self.window_size:]
            self._history[detector_name] = history

        if len(history) < self.min_consecutive:
            return None

        # Check consecutive exceedances
        recent = history[-self.min_consecutive:]
        if all(s > threshold for s in recent):
            mean_score = float(np.mean(recent))
            return Alert(
                detector=detector_name,
                score=mean_score,
                threshold=threshold,
                severity=Severity.WARNING,
                explanation=f"Rolling window alert: {self.min_consecutive} consecutive scores above threshold",
                metadata={"recent_scores": recent, "window_size": self.window_size},
            )

        return None


class AlertEngine:
    """Alert evaluation engine that coordinates multiple rules.

    Parameters
    ----------
    rules : list of BaseRule, optional
        List of alert rules to evaluate.
    """

    def __init__(self, rules: Optional[List[BaseRule]] = None):
        self.rules = rules or [ThresholdRule(), RollingWindowRule()]
        self._alerts: List[Alert] = []
        self._logger = get_logger("production_drift_detection.alert_engine")

    def evaluate(
        self,
        detector_name: str,
        score: float,
        threshold: float,
        **kwargs,
    ) -> List[Alert]:
        """Evaluate a score against all registered rules.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        score : float
            Current drift score.
        threshold : float
            Detector's base threshold.
        **kwargs
            Additional context passed to rules.

        Returns
        -------
        list of Alert
            All alerts triggered across rules.
        """
        alerts = []
        for rule in self.rules:
            try:
                alert = rule.evaluate(detector_name, score, threshold, **kwargs)
                if alert is not None:
                    alerts.append(alert)
                    self._alerts.append(alert)
                    self._logger.warning(f"Alert triggered: {alert}")
            except Exception as e:
                self._logger.error(f"Rule {type(rule).__name__} failed: {e}")

        return alerts

    def get_alerts(
        self,
        severity: Optional[Severity] = None,
        detector: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get filtered alerts.

        Parameters
        ----------
        severity : Severity, optional
            Filter by severity level.
        detector : str, optional
            Filter by detector name.
        limit : int, optional
            Maximum number of alerts to return, by default 100.

        Returns
        -------
        list of Alert
            Filtered alerts.
        """
        filtered = self._alerts
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        if detector:
            filtered = [a for a in filtered if a.detector == detector]
        return filtered[-limit:]

    def clear(self) -> None:
        """Clear all stored alerts."""
        self._alerts = []
        self._logger.info("Alert history cleared")
