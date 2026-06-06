#!/usr/bin/env python3
"""
DriftWatch — Comprehensive Benchmark Suite

Compares all detectors (KL, PSI, MMD, ADWIN) across:
  - Detection latency
  - False positive rate
  - Stability
  - Sensitivity to drift magnitude
  - Cross-magnitude comparison
  - Confidence-drift correlation effectiveness
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.detectors.adwin import ADWINDetector
from driftwatch.data.synthetic_drift import DriftGenerator
from driftwatch.evaluation.metrics import (
    compute_detection_latency,
    compute_false_positive_rate,
    compute_detection_stability,
    compute_sensitivity_to_drift,
    evaluate_detector,
)
from driftwatch.evaluation.benchmarks import BenchmarkFramework
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation


def print_separator(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_benchmark_1_comparison(framework: BenchmarkFramework) -> pd.DataFrame:
    """Benchmark 1: Standard comparison across all detectors."""
    print_separator("BENCHMARK 1: Detector Comparison (Covariate Shift, magnitude=1.0)")

    results = framework.run_benchmark(
        n_features=5,
        n_reference=1000,
        n_clean_batches=5,
        n_drift_batches=10,
        batch_size=100,
        drift_magnitude=1.0,
        drift_type="covariate",
    )

    print(f"\n{'Detector':<10} {'Threshold':<10} {'FPR':<10} {'Latency':<10} {'Detected?':<10} {'Stability':<10} {'Det Rate':<10}")
    print("-" * 70)
    for _, row in results.iterrows():
        lat = f"{row['detection_latency']}" if pd.notna(row['detection_latency']) else "N/A"
        det = "Yes" if row['drift_detected'] else "No"
        print(f"{row['detector']:<10} {row['threshold']:<10.4f} {row['fpr']:<10.2%} {lat:<10} {det:<10} {row['stability']:<10.4f} {row['detection_rate']:<10.2%}")

    return results


def run_benchmark_2_sensitivity(framework: BenchmarkFramework):
    """Benchmark 2: Sensitivity analysis across drift magnitudes."""
    print_separator("BENCHMARK 2: Sensitivity to Drift Magnitude")

    magnitudes = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    sensitivity_df = framework.run_sensitivity_analysis(magnitudes=magnitudes)

    # Pivot for clean reading
    pivot = sensitivity_df.pivot_table(
        index="magnitude",
        columns="detector",
        values="mean_score",
        aggfunc="mean",
    )
    print("\nMean Drift Score by Magnitude:\n")
    print(f"{'Magnitude':<12}", end="")
    for det in pivot.columns:
        print(f"{det:<16}", end="")
    print()
    print("-" * 12 + " " + "-" * (16 * len(pivot.columns)))
    for mag, row in pivot.iterrows():
        print(f"{mag:<12.2f}", end="")
        for val in row:
            print(f"{val:<16.6f}", end="")
        print()

    # Score increase ratio (score at max mag / score at zero mag)
    print(f"\n  Score Ratio (max / baseline):")
    for det in pivot.columns:
        if pivot.loc[0.0, det] > 0:
            ratio = pivot.loc[5.0, det] / max(pivot.loc[0.0, det], 1e-10)
            print(f"    {det:<12}: {ratio:.2f}x increase")
        else:
            print(f"    {det:<12}: (baseline near zero)")

    return sensitivity_df


def run_benchmark_3_drift_types():
    """Benchmark 3: Compare detectors across different drift types."""
    print_separator("BENCHMARK 3: Comparison Across Drift Types")

    drift_types = ["covariate", "perturbation"]
    detectors = {
        "kl": KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
        "psi": PSIDetector(threshold=0.1, n_bins=10),
        "mmd": MMDDetector(threshold=0.05, subsample=200),
    }

    all_rows = []
    for drift_type in drift_types:
        framework = BenchmarkFramework(list(detectors.values()), random_state=42)
        results = framework.run_benchmark(
            drift_magnitude=2.0,
            drift_type=drift_type,
        )
        results["drift_type"] = drift_type
        all_rows.append(results)

    combined = pd.concat(all_rows, ignore_index=True)
    
    print(f"\n{'Drift Type':<18} {'Detector':<10} {'FPR':<10} {'Latency':<10} {'Stability':<10}")
    print("-" * 60)
    for _, row in combined.iterrows():
        lat = f"{row['detection_latency']}" if pd.notna(row['detection_latency']) else "N/A"
        print(f"{row['drift_type']:<18} {row['detector']:<10} {row['fpr']:<10.2%} {lat:<10} {row['stability']:<10.4f}")

    return combined


def run_benchmark_4_adwin_streaming():
    """Benchmark 4: ADWIN streaming drift detection specifically."""
    print_separator("BENCHMARK 4: ADWIN Streaming Detection")
    
    generator = DriftGenerator(n_features=1, n_reference=500, random_state=42)
    reference = generator.generate_reference()
    
    # Gradual drift scenario
    gradual_data = generator.gradual_drift(
        n_steps=20, n_per_step=50, start_magnitude=0.0, end_magnitude=3.0
    )
    
    adwin = ADWINDetector(threshold=0.1, delta=0.05)
    adwin.fit(reference)
    
    scores = []
    window_sizes = []
    for i in range(20):
        batch = gradual_data[i * 50:(i + 1) * 50]
        result = adwin.detect(batch)
        scores.append(result["score"])
        window_sizes.append(adwin.window_size)
    
    print(f"\n  ADWIN Gradual Drift Detection ({'Yes' if max(scores) > 0.1 else 'No'}):")
    print(f"  Mean score: {np.mean(scores):.4f}")
    print(f"  Max score:  {max(scores):.4f}")
    print(f"  Drift detected in {sum(1 for s in scores if s > 0.1)}/{len(scores)} batches")
    
    # Latency analysis
    latency = compute_detection_latency(np.array(scores), 0.1, drift_start_idx=5)
    print(f"  Detection latency (batches): {latency.get('detection_latency_batches', 'N/A')}")
    
    return scores


def run_benchmark_5_confidence_early_warning():
    """Benchmark 5: Confidence as an early warning signal."""
    print_separator("BENCHMARK 5: Confidence Early-Warning Effectiveness")
    
    n_batches = 30
    rng = np.random.default_rng(42)
    
    # Scenario: confidence degrades BEFORE drift becomes severe
    cm = ConfidenceMonitor()
    corr = ConfidenceDriftCorrelation(max_lag=5)
    
    print("\n  Simulating: Confidence degrades before drift becomes severe")
    
    rows = []
    for i in range(n_batches):
        # Drift score increases after batch 15, but confidence starts dropping after batch 10
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
        
        rows.append({"batch": i + 1, "confidence": confidence, "drift_score": drift_score})
    
    summary = corr.summary()
    
    df = pd.DataFrame(rows)
    # Print early batches vs late batches
    early_conf = df[df["batch"] <= 10]["confidence"].mean()
    mid_conf = df[(df["batch"] > 10) & (df["batch"] <= 20)]["confidence"].mean()
    late_conf = df[df["batch"] > 20]["confidence"].mean()
    
    print(f"\n  Confidence trend: {early_conf:.4f} -> {mid_conf:.4f} -> {late_conf:.4f}")
    print(f"  Drift trend:      {df[df['batch'] <= 10]['drift_score'].mean():.4f} "
          f"-> {df[(df['batch'] > 10) & (df['batch'] <= 20)]['drift_score'].mean():.4f} "
          f"-> {df[df['batch'] > 20]['drift_score'].mean():.4f}")
    print(f"\n  Early warning score:     {summary['early_warning_score']:.2f}/100")
    print(f"  Early warnings detected: {summary['n_early_warnings']}")
    print(f"  Confidence leads drift:  {summary.get('confidence_is_leading_indicator', 'N/A')}")
    
    # Cross-correlation
    cc = corr.compute_cross_correlation()
    if "max_cross_correlation" in cc:
        print(f"  Max cross-correlation:   {cc['max_cross_correlation']:.4f}")
        print(f"  Optimal lead/lag:        {cc.get('optimal_lag', 'N/A')}")
    
    return df


def run_benchmark_6_feature_level():
    """Benchmark 6: PSI per-feature analysis."""
    print_separator("BENCHMARK 6: PSI Per-Feature Drift Analysis")
    
    generator = DriftGenerator(n_features=6, n_reference=1000, random_state=42)
    reference = generator.generate_reference()
    
    psi = PSIDetector(threshold=0.1, n_bins=10, per_feature=True)
    psi.fit(reference)
    
    # Inject drift in only features 0, 1, 2
    batch = generator.feature_perturbation_drift(
        reference[:200], noise_std=2.0, n_corrupt_features=3
    )
    
    result = psi.detect(batch)
    
    print(f"\n  Aggregate drift score: {result['score']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")
    
    if "per_feature_scores" in result:
        print(f"\n  Feature-level breakdown:")
        print(f"  {'Feature':<10} {'Score':<12} {'Status':<12}")
        print(f"  {'-'*8:<10} {'-'*10:<12} {'-'*10:<12}")
        for feat, score in result["per_feature_scores"].items():
            status = "DRIFTED" if score > psi.threshold else "stable"
            print(f"  {feat:<10} {score:<12.4f} {status:<12}")
    
    # Rank shifted features
    if "per_feature_scores" in result:
        sorted_feats = sorted(result["per_feature_scores"].items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Feature drift ranking (highest to lowest):")
        for i, (feat, score) in enumerate(sorted_feats, 1):
            print(f"    {i}. {feat}: {score:.4f}")
    
    return result


def print_footer(benchmark_results: dict):
    """Print final summary."""
    print_separator("BENCHMARK SUMMARY")
    
    print(f"\n  Detectors compared: 4 (KL, PSI, MMD, ADWIN)")
    print(f"  Drift types tested: covariate, perturbation, gradual")
    print(f"  Metrics computed:   FPR, latency, stability, sensitivity, cross-correlation")
    print(f"  Total benchmarks:   6")
    print()
    
    # Aggregate FPR and stability from benchmark 1
    if "comparison" in benchmark_results and not benchmark_results["comparison"].empty:
        df = benchmark_results["comparison"]
        print(f"  {'Detector':<10} {'FPR (mean)':<12} {'Stability (mean)':<18} {'Detection Rate':<16}")
        print(f"  {'-'*8:<10} {'-'*10:<12} {'-'*16:<18} {'-'*14:<16}")
        for _, row in df.iterrows():
            det = row["detector"]
            fpr = row["fpr"]
            stab = row["stability"]
            dr = row["detection_rate"]
            print(f"  {det:<10} {fpr:<12.2%} {stab:<18.4f} {dr:<16.2%}")
    
    print()
    print("=" * 72)
    print("  Benchmarks complete.")
    print("=" * 72)


def main():
    print()
    print("=" * 72)
    print("  DriftWatch — Comprehensive Detector Benchmark Suite")
    print("=" * 72)
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  NumPy: {np.__version__}, Pandas: {pd.__version__}")
    print()

    # Build detectors
    detectors = [
        KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
        PSIDetector(threshold=0.1, n_bins=10),
        MMDDetector(threshold=0.05, subsample=200),
    ]

    # Framework
    framework = BenchmarkFramework(detectors, random_state=42)

    results = {}

    # Benchmark 1: Standard comparison
    results["comparison"] = run_benchmark_1_comparison(framework)

    # Benchmark 2: Sensitivity
    results["sensitivity"] = run_benchmark_2_sensitivity(framework)

    # Benchmark 3: Drift types
    results["drift_types"] = run_benchmark_3_drift_types()

    # Benchmark 4: ADWIN streaming
    results["adwin"] = run_benchmark_4_adwin_streaming()

    # Benchmark 5: Confidence early warning
    results["confidence"] = run_benchmark_5_confidence_early_warning()

    # Benchmark 6: PSI per-feature
    results["feature_level"] = run_benchmark_6_feature_level()

    # Footer
    print_footer(results)


if __name__ == "__main__":
    main()
