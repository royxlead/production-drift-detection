"""Alert schemas for ProductionDriftDetection."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class Severity(str, Enum):
    """Alert severity levels.

    Levels represent increasing urgency:
    - HEALTHY: No drift detected, system operating normally.
    - WATCH: Minor drift detected, continue monitoring.
    - WARNING: Notable drift, investigate potential causes.
    - CRITICAL: Significant drift, immediate action recommended.
    """

    HEALTHY = "healthy"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"

    def __ge__(self, other: "Severity") -> bool:
        levels = [Severity.HEALTHY, Severity.WATCH, Severity.WARNING, Severity.CRITICAL]
        return levels.index(self) >= levels.index(other)

    def __lt__(self, other: "Severity") -> bool:
        levels = [Severity.HEALTHY, Severity.WATCH, Severity.WARNING, Severity.CRITICAL]
        return levels.index(self) < levels.index(other)


@dataclass
class Alert:
    """Data class representing a drift alert.

    Parameters
    ----------
    detector : str
        Name of the detector that triggered the alert.
    score : float
        Drift score at detection time.
    threshold : float
        Threshold that was exceeded.
    timestamp : datetime, optional
        When the alert was triggered, by default current time.
    severity : Severity, optional
        Severity level of the alert, by default Severity.WARNING.
    explanation : str, optional
        Human-readable explanation of the alert.
    metadata : dict, optional
        Additional metadata for the alert.
    """

    detector: str
    score: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    severity: Severity = Severity.WARNING
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization.

        Returns
        -------
        dict
            Serializable dictionary representation.
        """
        return {
            "detector": self.detector,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create Alert from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of an alert.

        Returns
        -------
        Alert
            Reconstructed Alert object.
        """
        return cls(
            detector=data["detector"],
            score=data["score"],
            threshold=data["threshold"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=Severity(data["severity"]),
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.detector}: "
            f"score={self.score:.4f} (threshold={self.threshold:.4f})"
            f"{' — ' + self.explanation if self.explanation else ''}"
        )
