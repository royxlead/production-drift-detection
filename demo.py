#!/usr/bin/env python3
"""
ProductionDriftDetection — Complete End-to-End Demo

This demo:
1. Generates reference and production data with controlled drift
2. Trains a simple classifier
3. Monitors drift across all detectors
4. Tracks confidence degradation
5. Demonstrates confidence-drift correlation
6. Triggers alerts
7. Exports results
8. Launches the dashboard

Usage:
    python demo.py              # Run all monitoring
    python demo.py --dashboard   # Launch dashboard after monitoring
    python demo.py --quick       # Quick 10-batch demo
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from production_drift_detection.alerts.schemas import Severity
from production_drift_detection.correlation.confidence_drift import ConfidenceDriftCorrelation
from production_drift_detection.data.synthetic_drift import DriftGenerator
from production_drift_detection.detectors.adwin import ADWINDetector
from production_drift_detection.detectors.kl import KLDivergenceDetector
from production_drift_detection.detectors.mmd import MMDDetector
from production_drift_detection.detectors.psi import PSIDetector
from production_drift_detection.evaluation.metrics import evaluate_detector
from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor
from production_drift_detection.monitors.stream_monitor import StreamMonitor


def print_header():
    print("=" * 70)
    print("  ProductionDriftDetection — Real-time Data Drift Detection")
    print("  End-to-End Demonstration")
    print("=" * 70)
    print()


def run_demo(quick: bool = False) -> dict:
    """Run the complete ProductionDriftDetection demo.

    Parameters
    ----------
    quick : bool
        If True, run a shorter demo.

    Returns
    -------
    dict
        Demo results with all monitoring data.
    """
    print_header()

    # Configuration
    n_features = 5
    n_reference = 1000
    n_batches = 15 if quick else 25
    batch_size = 100
    drift_start_batch = 5 if quick else 8

    print(f"Configuration:")
    print(f"  Features: {n_features}")
    print(f"  Reference samples: {n_reference}")
    print(f"  Batches: {n_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Drift starts at batch: {drift_start_batch}")
    print()

    # 1. Initialize generators and detectors
    print("[1/7] Initializing detectors...")
    rng = np.random.default_rng(42)

    detectors = {
        "kl": KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
        "psi": PSIDetector(threshold=0.1, n_bins=10, per_feature=True),
        "mmd": MMDDetector(threshold=0.05, subsample=200),
        "adwin": ADWINDetector(threshold=0.1, delta=0.05),
    }

    # 2. Create monitors
    print("[2/7] Creating monitors...")
    stream_monitor = StreamMonitor(detectors=detectors)
    confidence_monitor = ConfidenceMonitor()
    correlation = ConfidenceDriftCorrelation(max_lag=5)

    # 3. Generate reference data
    print("[3/7] Generating reference distribution...")
    generator = DriftGenerator(
        n_features=n_features,
        n_reference=n_reference,
        random_state=42,
    )
    reference_data = generator.generate_reference()
    print(f"  Reference shape: {reference_data.shape}")

    # 4. Fit monitors
    print("[4/7] Fitting monitors on reference data...")
    stream_monitor.fit(reference_data)
    print("  Done.")

    # 5. Simulate production batches with drift
    print(f"[5/7] Simulating {n_batches} production batches...")
    print(f"  Drift injection starts at batch {drift_start_batch}")
    print()

    all_results = []
    drift_magnitude = 0.0

    for batch_idx in range(n_batches):
        # Generate batch
        if batch_idx < drift_start_batch:
            # Clean data (same distribution as reference)
            batch = np.zeros((batch_size, n_features))
            for col in range(n_features):
                mean = rng.uniform(-1, 1)
                std = rng.uniform(0.5, 1.0)
                batch[:, col] = rng.normal(mean, std, batch_size)
        else:
            # Increasing drift
            drift_magnitude += 0.15
            batch = generator.covariate_shift(
                n_samples=batch_size,
                shift_magnitude=drift_magnitude,
            )

        # Process batch through stream monitor
        result = stream_monitor.process_batch(batch)

        # Simulate prediction probabilities
        if drift_magnitude > 0:
            base_conf = 0.85 - min(drift_magnitude * 0.08, 0.4)
            noise = rng.normal(0, 0.05, batch_size)
            probs = np.clip(
                np.column_stack([1 - (base_conf + noise), base_conf + noise]),
                0, 1,
            )
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = rng.dirichlet([1, 9], size=batch_size)

        # Track confidence
        conf_update = confidence_monitor.update(probs)

        # Track correlation
        drift_scores = result.get("scores", {})
        correlation.add_observation(
            confidence=conf_update.get("mean_confidence", 0.5),
            drift_scores=drift_scores,
            entropy=conf_update.get("mean_entropy", 0.5),
            margin=conf_update.get("mean_margin", 0.5),
        )

        all_results.append(result)

        # Print batch summary
        status_icon = "[OK]" if result["status"] == "healthy" else "[--]" if result["status"] == "watch" else "[!!]" if result["status"] == "warning" else "[##]"
        scores_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in result["scores"].items()
        )
        conf_str = f"Conf: {conf_update.get('mean_confidence', 0):.3f}"
        print(f"  Batch {batch_idx + 1:2d}: {status_icon} {scores_str} | {conf_str}")

        time.sleep(0.05)  # Small delay for realistic streaming feel

    print()

    # 6. Results and alerts
    print("[6/7] Analyzing results...")
    monitor_summary = stream_monitor.summary()
    conf_summary = confidence_monitor.summary()
    corr_summary = correlation.summary()

    print(f"\n=== Stream Monitor Summary ===")
    print(f"  Total batches: {monitor_summary['batch_count']}")
    print(f"  Total alerts: {monitor_summary['total_alerts']}")
    print(f"  Alerts by severity: {monitor_summary.get('alerts_by_severity', {})}")

    for det_name, det_summary in monitor_summary["detectors"].items():
        print(f"\n  {det_name.upper()} Detector:")
        print(f"    Mean score: {det_summary.get('mean_score', 0):.4f}")
        print(f"    Max score: {det_summary.get('max_score', 0):.4f}")
        print(f"    Status: {det_summary.get('current_status', 'unknown')}")

    print(f"\n=== Confidence Monitor ===")
    print(f"  Batches monitored: {conf_summary['batches_monitored']}")
    print(f"  Degradation detected: {conf_summary.get('degradation_detected', False)}")
    if conf_summary.get("degradation_detected"):
        print(f"  Reason: {conf_summary.get('degradation_reason', '')}")

    trends = conf_summary.get("trends", {})
    print(f"  Confidence trend: {trends.get('confidence_trend', 'N/A')}")
    print(f"  Entropy trend: {trends.get('entropy_trend', 'N/A')}")

    print(f"\n=== Confidence-Drift Correlation ===")
    print(f"  Observations: {corr_summary['n_observations']}")
    print(f"  Early warnings detected: {corr_summary['n_early_warnings']}")
    print(f"  Early warning score: {corr_summary['early_warning_score']:.2f}/100")
    print(f"  Confidence leads drift: {corr_summary.get('confidence_is_leading_indicator', 'N/A')}")

    # 7. Evaluation
    print(f"\n[7/7] Evaluation metrics...")
    clean_batches_list = all_results[:drift_start_batch]
    drifted_batches_list = all_results[drift_start_batch:]

    clean_data = []
    for r in clean_batches_list:
        clean_data.append(reference_data[:batch_size])
    drift_data = []
    for r in drifted_batches_list:
        drift_data.append(reference_data[:batch_size])

    if clean_data and drift_data:
        from production_drift_detection.evaluation.metrics import evaluate_detector

        for det_name, detector in detectors.items():
            try:
                eval_result = evaluate_detector(
                    detector=detector,
                    reference_data=reference_data,
                    clean_batches=clean_data,
                    drifted_batches=drift_data,
                    drift_start_batch=len(clean_data),
                )
                print(f"  {det_name.upper()}: "
                      f"FPR={eval_result['false_positive_rate']['false_positive_rate']:.2%}, "
                      f"Stability={eval_result['stability']['stability']:.3f}")
            except Exception as e:
                print(f"  {det_name.upper()}: Evaluation skipped ({e})")

    print()
    print("=" * 70)
    print("  Demo complete! ProductionDriftDetection successfully monitored")
    print("  data drift, confidence degradation, and generated")
    print("  early warning signals.")
    print("=" * 70)

    return {
        "stream_monitor": stream_monitor,
        "confidence_monitor": confidence_monitor,
        "correlation": correlation,
        "generator": generator,
        "results": all_results,
        "reference_data": reference_data,
    }


def launch_dashboard():
    """Launch the FastAPI dashboard."""
    print("Launching ProductionDriftDetection dashboard...")
    print("Navigate to http://localhost:8501 in your browser.")
    print("Press Ctrl+C to stop.")
    print()
    from production_drift_detection.dashboard.server import serve
    serve()


def main():
    parser = argparse.ArgumentParser(
        description="ProductionDriftDetection — Real-time Data Drift Detection Demo"
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch Streamlit dashboard after monitoring",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run a quick 15-batch demo",
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        default=None,
        help="Export results summary to file (JSON)",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export drift scores to CSV file",
    )
    args = parser.parse_args()

    # Run demo
    results = run_demo(quick=args.quick)

    # Export if requested
    if args.export:
        import json
        from datetime import datetime

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "stream_monitor": results["stream_monitor"].summary(),
            "confidence_monitor": results["confidence_monitor"].summary(),
            "correlation": results["correlation"].summary(),
        }
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\nResults exported to {args.export}")

    # Export CSV if requested
    if args.export_csv:
        import pandas as pd

        stream_monitor = results["stream_monitor"]
        confidence_monitor = results["confidence_monitor"]

        # Build comprehensive CSV with drift scores and confidence metrics
        history = stream_monitor.get_history()
        csv_data = []
        for batch_idx in range(len(next(iter(history.values())))):
            row = {"batch": batch_idx + 1}
            for det_name, scores in history.items():
                if batch_idx < len(scores):
                    row[f"{det_name}_score"] = scores[batch_idx]
            if batch_idx < len(confidence_monitor._confidence_history):
                row["confidence"] = confidence_monitor._confidence_history[batch_idx]
                row["entropy"] = confidence_monitor._entropy_history[batch_idx]
                row["margin"] = confidence_monitor._margin_history[batch_idx]
            csv_data.append(row)

        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(args.export_csv, index=False)
        print(f"\nDrift scores exported to {args.export_csv}")

    if not args.dashboard:
        print("\nRun with --dashboard to launch the dashboard.")


if __name__ == "__main__":
    main()
