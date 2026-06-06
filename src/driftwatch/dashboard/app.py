"""
DEPRECATED — Streamlit dashboard.

This file is kept for reference only. The dashboard has been replaced by
a FastAPI + vanilla JS frontend in `server.py` and the `static/` directory.

Launch the new dashboard:
    python -m driftwatch.dashboard.server

Or from Python:
    from driftwatch.dashboard import run_dashboard
    run_dashboard()
"""

import warnings

warnings.warn(
    "Streamlit dashboard is deprecated. Use `driftwatch.dashboard.server` instead.\n"
    "Run: python -m driftwatch.dashboard.server",
    DeprecationWarning,
    stacklevel=2,
)
