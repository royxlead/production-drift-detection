"""
DEPRECATED — Streamlit dashboard.

This file is kept for reference only. The dashboard has been replaced by
a FastAPI + vanilla JS frontend in `server.py` and the `static/` directory.

Launch the new dashboard:
    python -m production_drift_detection.dashboard.server

Or from Python:
    from production_drift_detection.dashboard import run_dashboard
    run_dashboard()
"""

import warnings

warnings.warn(
    "Streamlit dashboard is deprecated. Use `production_drift_detection.dashboard.server` instead.\n"
    "Run: python -m production_drift_detection.dashboard.server",
    DeprecationWarning,
    stacklevel=2,
)
