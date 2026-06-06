"""Dashboard package.

Provides a FastAPI-based interactive dashboard for drift monitoring.

Launch with:
    python -m driftwatch.dashboard.server
"""

from driftwatch.dashboard.visuals import Visualizer
from driftwatch.dashboard.server import serve as run_dashboard, app

__all__ = [
    "Visualizer",
    "run_dashboard",
    "app",
]
