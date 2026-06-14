"""
ProductionDriftDetection — Real-time data drift detection for deployed machine learning systems.

Detect distribution shifts in production data, monitor model confidence,
and trigger early warnings before model performance degrades.
"""

__version__ = "0.1.0"

from production_drift_detection.detectors.base import BaseDetector
from production_drift_detection.detectors.kl import KLDivergenceDetector
from production_drift_detection.detectors.psi import PSIDetector
from production_drift_detection.detectors.mmd import MMDDetector
from production_drift_detection.detectors.adwin import ADWINDetector

from production_drift_detection.monitors.stream_monitor import StreamMonitor
from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor

from production_drift_detection.alerts.schemas import Alert, Severity
from production_drift_detection.alerts.rules import AlertEngine, ThresholdRule, RollingWindowRule

from production_drift_detection.data.synthetic_drift import DriftGenerator
from production_drift_detection.data.loaders import DataLoader

from production_drift_detection.correlation.confidence_drift import ConfidenceDriftCorrelation

__all__ = [
    # Detectors
    "BaseDetector",
    "KLDivergenceDetector",
    "PSIDetector",
    "MMDDetector",
    "ADWINDetector",
    # Monitors
    "StreamMonitor",
    "ConfidenceMonitor",
    # Alerts
    "Alert",
    "Severity",
    "AlertEngine",
    "ThresholdRule",
    "RollingWindowRule",
    # Data
    "DriftGenerator",
    "DataLoader",
    # Correlation
    "ConfidenceDriftCorrelation",
    # Version
    "__version__",
]
