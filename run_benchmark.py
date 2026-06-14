#!/usr/bin/env python3
"""
ProductionDriftDetection — Focused Benchmark: Detection Latency, FPR & Sensitivity

Runs all detectors (KL, PSI, MMD, ADWIN) across multiple drift scenarios
and saves the results to a timestamped .txt file.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from production_drift_detection.detectors.kl import KLDivergenceDetector
from production_drift_detection.detectors.psi import PSIDetector
from production_drift_detection.detectors.mmd import MMDDetector
from production_drift_detection.detectors.adwin import ADWINDetector
from production_drift_detection.data.synthetic_drift import DriftGenerator
from production_drift_detection.evaluation.metrics import (
    compute_detection_latency,
    compute_false_positive_rate,
    compute_detection_stability,
    compute_sensitivity_to_drift,
    evaluate_detector,
)
from production_drift_detection.evaluation.benchmarks import BenchmarkFramework


def fmt(val, decimals=6):
    """Format a value for the txt output."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def run_latency_fpr_benchmark(output_path: str):
    start_time = time.time()

    # ── Detectors ──────────────────────────────────────────────────
    detectors = [
        ("KLDivergence", KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20)),
        ("PSI",           PSIDetector(threshold=0.1, n_bins=10)),
        ("MMD",           MMDDetector(threshold=0.05, subsample=200)),
        ("ADWIN",         ADWINDetector(threshold=0.1, delta=0.05)),
    ]

    rng = np.random.default_rng(42)
    generator = DriftGenerator(n_features=5, random_state=42)
    reference = generator.generate_reference()

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 80)
    w("  ProductionDriftDetection — Detection Latency, FPR & Sensitivity Benchmark")
    w("=" * 80)
    w(f"  Started:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  NumPy:         {np.__version__}")
    w(f"  Pandas:        {pd.__version__}")
    w(f"  Detectors:     {', '.join(d[0] for d in detectors)}")
    w(f"  Features:      5")
    w(f"  Reference:     1000 samples")
    w(f"  Batch size:    100")
    w(f"  Clean batches: 5  |  Drifted batches: 10")
    w("=" * 80)

    # ── 1. Detection Latency & FPR per Detector ────────────────────
    w("\n" + "=" * 80)
    w("  1. DETECTION LATENCY & FALSE POSITIVE RATE")
    w("=" * 80)

    framework = BenchmarkFramework([d[1] for d in detectors], random_state=42)

    # Run benchmark at multiple drift magnitudes
    for magnitude in [0.5, 1.0, 2.0, 3.0]:
        w(f"\n  ── Drift magnitude: {magnitude:.1f} ──")
        w(f"  {'Detector':<14} {'Threshold':<10} {'FPR':<12} {'Latency (batches)':<18} "
          f"{'Detected?':<12} {'Detection Rate':<16} {'Stability':<12}")
        w("  " + "-" * 94)

        results = framework.run_benchmark(
            n_features=5, n_reference=1000,
            n_clean_batches=5, n_drift_batches=10,
            batch_size=100, drift_magnitude=magnitude,
            drift_type="covariate",
        )

        for _, row in results.iterrows():
            lat = fmt(row.get("detection_latency"))
            fpr_v = fmt(row.get("fpr"), 4)
            det = fmt(row.get("drift_detected"))
            dr = fmt(row.get("detection_rate"), 2)
            stab = fmt(row.get("stability"), 4)
            thresh = fmt(row.get("threshold"), 4)
            w(f"  {row['detector']:<14} {thresh:<10} {fpr_v:<12} {lat:<18} "
              f"{det:<12} {dr:<16} {stab:<12}")

    # ── 2. SENSITIVITY ANALYSIS ────────────────────────────────────
    w("\n" + "=" * 80)
    w("  2. SENSITIVITY TO DRIFT MAGNITUDE")
    w("=" * 80)

    magnitudes = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    sensitivity_df = framework.run_sensitivity_analysis(magnitudes=magnitudes)

    # Per-detector sensitivity table
    pivot = sensitivity_df.pivot_table(
        index="magnitude", columns="detector", values="mean_score", aggfunc="mean"
    )
    w(f"\n  Mean Drift Score by Magnitude:\n")
    header = f"  {'Magnitude':<12}"
    for det in pivot.columns:
        header += f"{det:<16}"
    w(header)
    w("  " + "-" * 12 + " " + "-" * (16 * len(pivot.columns)))
    for mag, row in pivot.iterrows():
        line = f"  {mag:<12.2f}"
        for val in row:
            line += f"{val:<16.6f}"
        w(line)

    w(f"\n  Score Ratio (max magnitude / baseline):")
    for det in pivot.columns:
        baseline = max(pivot.loc[0.0, det], 1e-10)
        ratio = pivot.loc[5.0, det] / baseline
        w(f"    {det:<14}: {ratio:.2f}x")

    # Sensitivity slope
    w(f"\n  Sensitivity (score increase per unit magnitude):")
    for det in pivot.columns:
        mags = np.array(magnitudes)
        scores = pivot[det].values
        slope = np.polyfit(mags, scores, 1)[0]
        w(f"    {det:<14}: {slope:.6f}")

    # ── 3. DETECTOR-LEVEL EVALUATION SUMMARY ───────────────────────
    w("\n" + "=" * 80)
    w("  3. COMPREHENSIVE DETECTOR EVALUATION (magnitude=2.0)")
    w("=" * 80)

    data = framework._generate_benchmark_data(
        n_features=5, n_reference=1000,
        n_clean_batches=5, n_drift_batches=10,
        batch_size=100, drift_magnitude=2.0, drift_type="covariate",
    )

    for det_name, detector in detectors:
        detector.fit(data["reference"])
        all_scores = []
        for batch in data["clean_batches"]:
            all_scores.append(detector.score(batch))
        for batch in data["drifted_batches"]:
            all_scores.append(detector.score(batch))

        scores_arr = np.array(all_scores)

        fpr_metrics = compute_false_positive_rate(scores_arr, detector.threshold, data["drift_start_batch"])
        latency_metrics = compute_detection_latency(scores_arr, detector.threshold, data["drift_start_batch"])
        stability_metrics = compute_detection_stability(scores_arr)

        w(f"\n  ── {det_name} ──")
        w(f"    Threshold:                       {detector.threshold:.4f}")
        w(f"    FPR (before drift):              {fpr_metrics['false_positive_rate']:.4%}  "
          f"({fpr_metrics['false_positives']}/{fpr_metrics['total_pre_drift']} pre-drift batches)")
        w(f"    Detection latency (batches):     {latency_metrics.get('detection_latency_batches', 'N/A')}")
        w(f"    Detection latency (samples):     {latency_metrics.get('detection_latency_samples', 'N/A')}")
        w(f"    Drift detected:                  {latency_metrics.get('drift_detected', False)}")
        w(f"    Detection rate:                  {latency_metrics.get('detection_rate', 0):.2%}")
        w(f"    Stability:                       {stability_metrics['stability']:.4f}")
        w(f"    Score mean / std:                {stability_metrics['score_mean']:.6f} / {stability_metrics['score_std']:.6f}")
        w(f"    Autocorrelation:                 {stability_metrics.get('autocorrelation', 0):.4f}")
        w(f"    Coefficient of variation:        {stability_metrics.get('coefficient_of_variation', 0):.4f}")

    # ── 4. DRIFT TYPE COMPARISON ───────────────────────────────────
    w("\n" + "=" * 80)
    w("  4. CROSS-DRIFT-TYPE COMPARISON (magnitude=2.0)")
    w("=" * 80)

    for drift_type in ["covariate", "perturbation"]:
        w(f"\n  ── Drift type: {drift_type} ──")
        ft = BenchmarkFramework([d[1] for d in detectors], random_state=42)
        res = ft.run_benchmark(drift_magnitude=2.0, drift_type=drift_type)

        w(f"  {'Detector':<14} {'FPR':<12} {'Latency (batches)':<18} {'Detected?':<12} {'Stability':<12} {'Det Rate':<12}")
        w("  " + "-" * 80)
        for _, row in res.iterrows():
            w(f"  {row['detector']:<14} {fmt(row['fpr'], 4):<12} "
              f"{fmt(row.get('detection_latency')):<18} {fmt(row.get('drift_detected')):<12} "
              f"{fmt(row.get('stability'), 4):<12} {fmt(row.get('detection_rate'), 2):<12}")

    # ── 5. ADWIN STREAMING BENCHMARK ───────────────────────────────
    w("\n" + "=" * 80)
    w("  5. ADWIN STREAMING DETECTION (Gradual Drift)")
    w("=" * 80)

    gen = DriftGenerator(n_features=1, n_reference=500, random_state=42)
    ref = gen.generate_reference()
    gradual_data = gen.gradual_drift(n_steps=20, n_per_step=50, start_magnitude=0.0, end_magnitude=3.0)

    adwin = ADWINDetector(threshold=0.1, delta=0.05)
    adwin.fit(ref)

    scores = []
    for i in range(20):
        batch = gradual_data[i * 50:(i + 1) * 50]
        result = adwin.detect(batch)
        scores.append(result["score"])

    scores_arr = np.array(scores)
    adwin_latency = compute_detection_latency(scores_arr, 0.1, drift_start_idx=5)
    adwin_fpr = compute_false_positive_rate(scores_arr, 0.1, drift_start_idx=5)

    w(f"\n    Mean ADWIN score:           {np.mean(scores):.4f}")
    w(f"    Max ADWIN score:            {max(scores):.4f}")
    w(f"    Drift detected in:          {sum(1 for s in scores if s > 0.1)}/{len(scores)} batches")
    w(f"    Detection latency:          {adwin_latency.get('detection_latency_batches', 'N/A')} batches")
    w(f"    FPR (pre-drift):            {adwin_fpr['false_positive_rate']:.4%}")

    # ── 6. CONFIDENCE EARLY WARNING ────────────────────────────────
    w("\n" + "=" * 80)
    w("  6. CONFIDENCE EARLY WARNING EFFECTIVENESS")
    w("=" * 80)

    from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor
    from production_drift_detection.correlation.confidence_drift import ConfidenceDriftCorrelation

    cm = ConfidenceMonitor()
    corr = ConfidenceDriftCorrelation(max_lag=5)

    n_batches = 30
    for i in range(n_batches):
        if i < 10:
            confidence = 0.92 - rng.normal(0, 0.02)
            drift_score = 0.02 + rng.normal(0, 0.01)
        elif i < 15:
            confidence = 0.88 - (i - 10) * 0.015 + rng.normal(0, 0.02)
            drift_score = 0.03 + rng.normal(0, 0.01)
        else:
            confidence = 0.78 - (i - 15) * 0.02 + rng.normal(0, 0.02)
            drift_score = 0.05 + (i - 15) * 0.03 + rng.normal(0, 0.02)

        confidence = max(0.1, min(0.99, confidence))
        drift_score = max(0.0, drift_score)
        probs = np.array([[1 - confidence, confidence]])
        cm.update(probs)
        corr.add_observation(
            confidence=confidence,
            drift_scores={"psi": drift_score, "kl": drift_score * 0.8, "mmd": drift_score * 0.4},
            entropy=-confidence * np.log(confidence) - (1 - confidence) * np.log(1 - confidence) if 0 < confidence < 1 else 0,
            margin=abs(2 * confidence - 1),
        )

    summary = corr.summary()
    w(f"\n    Early warning score:       {summary['early_warning_score']:.2f}/100")
    w(f"    Early warnings detected:   {summary['n_early_warnings']}")
    w(f"    Confidence leads drift:    {summary.get('confidence_is_leading_indicator', 'N/A')}")

    cc = corr.compute_cross_correlation()
    if "max_cross_correlation" in cc:
        w(f"    Max cross-correlation:     {cc['max_cross_correlation']:.4f}")
        w(f"    Optimal lead/lag:          {cc.get('optimal_lag', 'N/A')}")

    # ── SUMMARY TABLE ──────────────────────────────────────────────
    w("\n" + "=" * 80)
    w("  BENCHMARK SUMMARY")
    w("=" * 80)
    w(f"\n  {'Detector':<14} {'FPR (mag=1.0)':<16} {'FPR (mag=2.0)':<16} "
      f"{'Latency (mag=1.0)':<18} {'Latency (mag=2.0)':<18} {'Sensitivity':<14}")
    w("  " + "-" * 96)

    for mag_small, mag_large in [(1.0, 2.0)]:
        res_small = framework.run_benchmark(drift_magnitude=mag_small)
        res_large = framework.run_benchmark(drift_magnitude=mag_large)

        for _, row_s in res_small.iterrows():
            det = row_s["detector"]
            row_l = res_large[res_large["detector"] == det]
            fpr_s = fmt(row_s["fpr"], 4)
            fpr_l = fmt(row_l["fpr"].values[0], 4) if len(row_l) > 0 else "N/A"
            lat_s = fmt(row_s.get("detection_latency"))
            lat_l = fmt(row_l["detection_latency"].values[0]) if len(row_l) > 0 else "N/A"

            # Compute sensitivity
            if det in pivot.columns:
                sens = fmt(np.polyfit(magnitudes, pivot[det].values, 1)[0], 6)
            else:
                sens = "N/A"

            w(f"  {det:<14} {fpr_s:<16} {fpr_l:<16} {lat_s:<18} {lat_l:<18} {sens:<14}")

    elapsed = time.time() - start_time
    w(f"\n  Total benchmark time: {elapsed:.2f} seconds")
    w(f"  Benchmark completed:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 80)

    # ── Write to file ──────────────────────────────────────────────
    output = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    # Also print to console (handle non-UTF-8 consoles)
    try:
        print(output)
        print(f"\n  Results saved to: {output_path}")
    except UnicodeEncodeError:
        # Fallback for Windows codepage consoles
        safe = output.encode("ascii", errors="replace").decode("ascii")
        print(safe)
        print(f"\n  Results saved to: {output_path}")
    return output


if __name__ == "__main__":
    output_path = Path(__file__).parent / "benchmark_results.txt"
    run_latency_fpr_benchmark(str(output_path))
