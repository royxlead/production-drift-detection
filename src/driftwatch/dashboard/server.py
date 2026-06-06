"""FastAPI backend server for the DriftWatch dashboard.

Provides a REST API for drift monitoring data. Serves the static
frontend and exposes data endpoints for all 6 dashboard pages.

Launch with:
    python -m driftwatch.dashboard.server
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from driftwatch.alerts.schemas import Severity
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation
from driftwatch.data.synthetic_drift import DriftGenerator
from driftwatch.detectors.adwin import ADWINDetector
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.monitors.stream_monitor import StreamMonitor

# ---------------------------------------------------------------------------
# Data initialization
# ---------------------------------------------------------------------------

app = FastAPI(title="DriftWatch Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals populated on first request
_stream_monitor: Optional[StreamMonitor] = None
_confidence_monitor: Optional[ConfidenceMonitor] = None
_correlation: Optional[ConfidenceDriftCorrelation] = None
_generator: Optional[DriftGenerator] = None
_results_log: List[Dict[str, Any]] = []
_n_batches = 25


def _initialize() -> None:
    """Generate sample data and fit monitors (lazy init on first request)."""
    global _stream_monitor, _confidence_monitor, _correlation, _generator, _results_log

    if _stream_monitor is not None:
        return

    n_features = 5
    n_reference = 1000
    batch_size = 100
    drift_start_batch = 8
    rng = np.random.default_rng(42)

    # Detectors
    detectors = {
        "kl": KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
        "psi": PSIDetector(threshold=0.1, n_bins=10, per_feature=True),
        "mmd": MMDDetector(threshold=0.05, subsample=200),
        "adwin": ADWINDetector(threshold=0.1, delta=0.05),
    }

    _stream_monitor = StreamMonitor(detectors=detectors)
    _confidence_monitor = ConfidenceMonitor()
    _correlation = ConfidenceDriftCorrelation(max_lag=5)
    _generator = DriftGenerator(n_features=n_features, n_reference=n_reference, random_state=42)

    reference = _generator.generate_reference()
    _stream_monitor.fit(reference)

    drift_magnitude = 0.0
    for batch_idx in range(_n_batches):
        if batch_idx < drift_start_batch:
            batch = np.zeros((batch_size, n_features))
            for col in range(n_features):
                mean = rng.uniform(-1, 1)
                std = rng.uniform(0.5, 1.0)
                batch[:, col] = rng.normal(mean, std, batch_size)
        else:
            drift_magnitude += 0.15
            batch = _generator.covariate_shift(n_samples=batch_size, shift_magnitude=drift_magnitude)

        result = _stream_monitor.process_batch(batch)

        if drift_magnitude > 0:
            base_conf = 0.85 - min(drift_magnitude * 0.08, 0.4)
            noise = rng.normal(0, 0.05, batch_size)
            probs = np.clip(np.column_stack([1 - (base_conf + noise), base_conf + noise]), 0, 1)
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = rng.dirichlet([1, 9], size=batch_size)

        conf_update = _confidence_monitor.update(probs)
        drift_scores = result.get("scores", {})
        _correlation.add_observation(
            confidence=conf_update.get("mean_confidence", 0.5),
            drift_scores=drift_scores,
            entropy=conf_update.get("mean_entropy", 0.5),
            margin=conf_update.get("mean_margin", 0.5),
        )
        _results_log.append(result)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    _initialize()


@app.get("/api/overview")
def get_overview() -> Dict[str, Any]:
    sm = _stream_monitor
    cm = _confidence_monitor
    cr = _correlation
    summary = sm.summary()
    return {
        "status": summary.get("current_status", "healthy"),
        "active_detectors": len(sm.detectors),
        "batches_processed": summary.get("batch_count", 0),
        "total_alerts": summary.get("total_alerts", 0),
        "alerts_by_severity": summary.get("alerts_by_severity", {}),
        "early_warning_score": cr.compute_early_warning_score().get("early_warning_score", 0),
        "confidence_drop": (
            (cm._confidence_history[0] - cm._confidence_history[-1]) / max(cm._confidence_history[0], 1e-10) * 100
        ) if len(cm._confidence_history) > 1 else 0,
    }


@app.get("/api/drift-scores")
def get_drift_scores() -> Dict[str, Any]:
    history = _stream_monitor.get_history()
    return {
        "batches": list(range(1, max(len(v) for v in history.values()) + 1)),
        "detectors": history,
    }


@app.get("/api/detector-details/{detector_name}")
def get_detector_details(detector_name: str) -> Dict[str, Any]:
    det = _stream_monitor.detectors.get(detector_name)
    if det is None:
        return {"error": f"Unknown detector: {detector_name}"}
    summary = det.summary()
    history = _stream_monitor.get_history().get(detector_name, [])
    return {
        "name": detector_name,
        "summary": summary,
        "scores": history,
        "current_score": history[-1] if history else None,
        "mean_score": summary.get("mean_score", 0),
        "max_score": summary.get("max_score", 0),
        "num_alerts": summary.get("num_alerts", 0),
    }


@app.get("/api/feature-scores")
def get_feature_scores() -> Dict[str, Any]:
    psi = _stream_monitor.detectors.get("psi")
    if psi is None or not hasattr(psi, "get_feature_scores"):
        return {"error": "PSI detector not configured for per-feature analysis"}
    scores = psi.get_feature_scores()
    names = psi.feature_names or [f"feature_{i}" for i in range(len(scores[0]))] if scores else []
    return {"feature_names": names, "scores": scores}


@app.get("/api/confidence")
def get_confidence() -> Dict[str, Any]:
    cm = _confidence_monitor
    trends = cm.get_trends()
    uncertainty = cm.get_uncertainty_metrics()
    degraded, reason = cm.degradation_detected()
    cal = cm.get_calibration_summary()
    return {
        "confidence_history": cm._confidence_history,
        "entropy_history": cm._entropy_history,
        "margin_history": cm._margin_history,
        "trends": trends,
        "uncertainty": uncertainty,
        "degraded": degraded,
        "degradation_reason": reason,
        "calibration": cal,
    }


@app.get("/api/correlation")
def get_correlation() -> Dict[str, Any]:
    cr = _correlation
    viz = cr.get_visualization_data()
    score = cr.compute_early_warning_score()
    cross = {}
    for det_name in cr._drift_history:
        cross[det_name] = cr.compute_cross_correlation(drift_key=det_name)
    summary = cr.summary()
    return {
        "visualization": viz,
        "early_warning_score": score,
        "cross_correlations": cross,
        "summary": summary,
    }


@app.get("/api/alerts")
def get_alerts(
    severity: Optional[str] = Query(None),
    detector: Optional[str] = Query(None),
    limit: int = Query(100),
) -> List[Dict[str, Any]]:
    sev = None
    if severity and severity != "all":
        try:
            sev = Severity(severity)
        except ValueError:
            pass
    alerts = _stream_monitor.get_alerts(severity=sev, limit=limit)
    if detector:
        alerts = [a for a in alerts if a.detector == detector]
    return [a.to_dict() for a in alerts]


# ---------------------------------------------------------------------------
# Static file serving & startup
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def serve(host: str = "127.0.0.1", port: int = 8501) -> None:
    """Launch the dashboard server."""
    import uvicorn
    print(f"DriftWatch dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
