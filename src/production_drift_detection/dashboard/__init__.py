"""Dashboard package.

Provides a FastAPI-based interactive dashboard for drift monitoring.

Launch with:
    python -m production_drift_detection.dashboard.server
"""

from production_drift_detection.dashboard.visuals import Visualizer
from production_drift_detection.dashboard.server import serve as run_dashboard, app

__all__ = [
    "Visualizer",
    "run_dashboard",
    "app",
]
