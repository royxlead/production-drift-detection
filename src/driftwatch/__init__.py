"""
DriftWatch — Real-time data drift detection for deployed machine learning systems.

Detect distribution shifts in production data, monitor model confidence,
and trigger early warnings before model performance degrades.
"""

__version__ = "0.1.0"

from driftwatch.detectors.base import BaseDetector
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.detectors.adwin import ADWINDetector

from driftwatch.monitors.stream_monitor import StreamMonitor
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor

from driftwatch.alerts.schemas import Alert, Severity
from driftwatch.alerts.rules import AlertEngine, ThresholdRule, RollingWindowRule

from driftwatch.data.synthetic_drift import DriftGenerator
from driftwatch.data.loaders import DataLoader

from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation

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
