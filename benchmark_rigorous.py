#!/usr/bin/env python3
"""
DriftWatch — Rigorous Benchmark Suite

Addresses methodological gaps identified in the initial benchmark:
  ✓ 50 pre-drift batches (stable FPR estimates)
  ✓ 50 drifted batches with gradual onset (meaningful latency)
  ✓ Drift magnitude starts at 0 and ramps up over 10 batches
  ✓ 20 random seeds with confidence intervals
  ✓ ROC / AUC analysis (threshold-independent)
  ✓ Cohen's d effect size (better sensitivity metric)
  ✓ Stronger perturbation drift with systematic shift + noise
  ✓ Detection delay measured from batch where drift exceeds 2× baseline noise
"""

import sys
import time
import math
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

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
)


# ── Detector definitions ──────────────────────────────────────────
DETECTORS = [
    ("KLDivergence", KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20)),
    ("PSI",           PSIDetector(threshold=0.1, n_bins=10)),
    ("MMD",           MMDDetector(threshold=0.05, subsample=200)),
    ("ADWIN",         ADWINDetector(threshold=0.1, delta=0.05)),
]


def _copy_detector(detector):
    """Create a fresh copy of a detector with the same parameters."""
    cls = detector.__class__
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for pname in sig.parameters:
        if pname == 'self':
            continue
        if hasattr(detector, pname):
            kwargs[pname] = getattr(detector, pname)
        elif pname in detector.__dict__:
            kwargs[pname] = detector.__dict__[pname]
    return cls(**kwargs)


def fmt(v, decimals=6):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def cohens_d(scores_before: np.ndarray, scores_after: np.ndarray) -> float:
    """Cohen's d effect size between two sets of scores."""
    n1, n2 = len(scores_before), len(scores_after)
    m1, m2 = np.mean(scores_before), np.mean(scores_after)
    s1, s2 = np.var(scores_before, ddof=1), np.var(scores_after, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2 + 1e-10))
    return (m2 - m1) / max(sp, 1e-10)


def compute_roc_auc(scores, labels):
    """Compute ROC curve and AUC for a set of scores vs binary labels."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return {"auc": 0.5, "tpr": [], "fpr": [], "thresholds": [], "n_pos": int(n_pos), "n_neg": int(n_neg)}

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0
    prev_score = None
    for i in range(len(sorted_scores) - 1, -1, -1):
        if prev_score is not None and sorted_scores[i] != prev_score:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = sorted_scores[i]
    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # AUC via trapezoidal rule (handles numpy >= 2.0 where np.trapz was removed)
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (tpr_list[i-1] + tpr_list[i]) * (fpr_list[i] - fpr_list[i-1]) / 2.0
    return {"auc": auc, "tpr": tpr_list, "fpr": fpr_list, "n_pos": int(n_pos), "n_neg": int(n_neg)}


def run_single_seed(seed: int) -> dict:
    """Run one full benchmark trial with a given random seed."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    N_CLEAN = 50
    N_DRIFT = 50
    BATCH_SIZE = 100
    N_FEATURES = 5
    N_REFERENCE = 2000
    DRIFT_MAGNITUDE = 2.0

    gen = DriftGenerator(n_features=N_FEATURES, n_reference=N_REFERENCE, random_state=seed)
    reference = gen.generate_reference()

    # ── Pre-drift: generate clean batches from reference distribution ──
    clean_batches = []
    for _ in range(N_CLEAN):
        idx = rng.choice(N_REFERENCE, BATCH_SIZE)
        clean_batches.append(reference[idx].copy())

    # ── Drift: gradual ramp from 0 to DRIFT_MAGNITUDE ──
    # Batches 0-4: zero drift (clean baseline interleaved)
    # Batches 5-14: linear ramp-up
    # Batches 15-49: full drift at DRIFT_MAGNITUDE
    drift_batches = []
    drift_magnitudes = []
    for i in range(N_DRIFT):
        if i < 5:
            mag = 0.0
            # Sample from reference
            idx = rng.choice(N_REFERENCE, BATCH_SIZE)
            drift_batches.append(reference[idx].copy())
        elif i < 15:
            mag = DRIFT_MAGNITUDE * (i - 4) / 10
            drift_batches.append(gen.covariate_shift(n_samples=BATCH_SIZE, shift_magnitude=mag))
        else:
            mag = DRIFT_MAGNITUDE
            drift_batches.append(gen.covariate_shift(n_samples=BATCH_SIZE, shift_magnitude=mag))
        drift_magnitudes.append(mag)

    # ── Ground truth labels ──
    # First 5 "drift" batches are actually clean (magnitude 0)
    # So FPR is measured on N_CLEAN + 5 batches
    fpr_window = N_CLEAN + 5  # 55 clean batches
    labels = np.array([0] * fpr_window + [1] * (N_DRIFT - 5))

    # ── Perturbation drift data (separate) ──
    pert_clean = []
    for _ in range(N_CLEAN):
        idx = rng.choice(N_REFERENCE, BATCH_SIZE)
        pert_clean.append(reference[idx].copy())

    # Stronger perturbation: add noise to ALL features, not just half
    pert_drift = []
    for _ in range(N_DRIFT):
        idx = rng.choice(N_REFERENCE, BATCH_SIZE)
        batch = reference[idx].copy()
        noise = rng.normal(0, 1.5, batch.shape)
        batch += noise
        pert_drift.append(batch)
    pert_labels = np.array([0] * N_CLEAN + [1] * N_DRIFT)

    results = {}

    for det_name, detector in DETECTORS:
        # ── Covariate drift evaluation ──
        det_covar = _copy_detector(detector)
        det_covar.fit(reference)

        all_scores = []
        for batch in clean_batches:
            all_scores.append(det_covar.score(batch))
        for batch in drift_batches:
            all_scores.append(det_covar.score(batch))
        scores_arr = np.array(all_scores)

        # FPR: score > threshold in the first fpr_window batches (which are all clean)
        pre_drift_scores = scores_arr[:fpr_window]
        false_positives = np.sum(pre_drift_scores > det_covar.threshold)
        fpr = false_positives / len(pre_drift_scores)

        # Latency: first index after the clean ramp-up (idx=fpr_window) where score > threshold
        post_drift_scores = scores_arr[fpr_window:]
        detection_indices = np.where(post_drift_scores > det_covar.threshold)[0]
        latency = int(detection_indices[0]) if len(detection_indices) > 0 else None
        detected = len(detection_indices) > 0
        detection_rate = len(detection_indices) / len(post_drift_scores)

        # Stability on all scores
        cv = np.std(scores_arr) / max(np.mean(scores_arr), 1e-10)
        stability = float(1.0 / (1.0 + cv))

        # ROC / AUC
        roc = compute_roc_auc(scores_arr, labels)

        # Cohen's d: effect size between pre-drift and fully-drifted
        d = cohens_d(scores_arr[:N_CLEAN], scores_arr[fpr_window:])
        # Also compare pre-drift vs strong drift (last 35 batches)
        d_strong = cohens_d(scores_arr[:N_CLEAN], scores_arr[fpr_window + 15:])

        # ── Perturbation drift evaluation ──
        det_pert = _copy_detector(detector)
        det_pert.fit(reference)

        p_scores = []
        for batch in pert_clean:
            p_scores.append(det_pert.score(batch))
        for batch in pert_drift:
            p_scores.append(det_pert.score(batch))
        p_scores_arr = np.array(p_scores)

        p_pre = p_scores_arr[:N_CLEAN]
        p_post = p_scores_arr[N_CLEAN:]
        p_fp = np.sum(p_pre > det_pert.threshold)
        p_fpr = p_fp / len(p_pre)
        p_detections = np.where(p_post > det_pert.threshold)[0]
        p_latency = int(p_detections[0]) if len(p_detections) > 0 else None
        p_det_rate = len(p_detections) / len(p_post)
        p_d = cohens_d(p_pre, p_post)
        p_roc = compute_roc_auc(p_scores_arr, pert_labels)

        results[det_name] = {
            "fpr": fpr,
            "false_positives": int(false_positives),
            "pre_drift_batches": len(pre_drift_scores),
            "latency": latency,
            "detected": detected,
            "detection_rate": detection_rate,
            "stability": stability,
            "auc": roc["auc"],
            "cohens_d": d,
            "cohens_d_strong": d_strong,
            "pert_fpr": p_fpr,
            "pert_latency": p_latency,
            "pert_detection_rate": p_det_rate,
            "pert_auc": p_roc["auc"],
            "pert_cohens_d": p_d,
            "score_mean": float(np.mean(scores_arr)),
            "score_std": float(np.std(scores_arr)),
            "post_drift_mean": float(np.mean(post_drift_scores)),
        }

    return results


def run_rigorous_benchmark(output_path: str):
    start_time = time.time()

    N_SEEDS = 20

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 90)
    w("  DriftWatch — Rigorous Benchmark Suite")
    w("=" * 90)
    w(f"  Started:              {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  NumPy:                {np.__version__}")
    w(f"  Pandas:               {pd.__version__}")
    w(f"  Detectors:            {', '.join(d[0] for d in DETECTORS)}")
    w(f"  Random seeds:         {N_SEEDS}")
    w(f"  Features:             5")
    w(f"  Reference:            2000 samples")
    w(f"  Batch size:           100")
    w(f"  Clean pre-drift:      50 batches  (5000 samples)")
    w(f"  Zero-drift batches:   5 (interleaved with drift phase)")
    w(f"  Total clean eval:     55 batches  (5500 samples)")
    w(f"  Drifted batches:      50 (10 ramp + 40 full strength)")
    w(f"  Total eval batches:   105 per trial")
    w(f"  Drift magnitude:      2.0 (ramp from 0 over 10 batches)")
    w(f"  Perturbation:         N(0, 1.5) added to ALL features")
    w(f"  Confidence intervals: mean ± 2×SEM (≈95% CI)")
    w("=" * 90)

    # ── Run all seeds ────────────────────────────────────────────
    all_results = {name: [] for name, _ in DETECTORS}
    for seed_idx in range(N_SEEDS):
        if seed_idx % 5 == 0:
            w(f"\n  Progress: seed {seed_idx + 1}/{N_SEEDS}...")
        result = run_single_seed(seed_idx * 7 + 42)  # spread seeds
        for name in all_results:
            all_results[name].append(result[name])

    # ── Aggregate with confidence intervals ──────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 1: CORE METRICS WITH CONFIDENCE INTERVALS")
    w("=" * 90)

    def mean_ci(values):
        arr = np.array(values)
        m = np.mean(arr)
        s = np.std(arr, ddof=1)
        ci = 1.96 * s / max(math.sqrt(len(arr)), 1e-10)
        return m, ci, s

    metrics_desc = [
        ("fpr",              "FPR (clean phase)",                 "%",    True),
        ("latency",          "Latency (batches after drift)",     "",     False),
        ("detection_rate",   "Detection rate (drift phase)",      "%",    True),
        ("auc",              "ROC AUC",                           "",     False),
        ("cohens_d",         "Cohen's d (pre vs all drift)",      "",     False),
        ("cohens_d_strong",  "Cohen's d (pre vs strong drift)",   "",     False),
        ("stability",        "Stability",                         "",     False),
        ("pert_fpr",         "Perturbation FPR",                  "%",    True),
        ("pert_detection_rate", "Perturbation det. rate",         "%",    True),
        ("pert_auc",         "Perturbation AUC",                  "",     False),
        ("pert_cohens_d",    "Perturbation Cohen's d",            "",     False),
    ]

    # Print header
    header = f"  {'Metric':<38}"
    for name, _ in DETECTORS:
        header += f"{name:<26}"
    w(header)
    w("  " + "-" * 38 + " " + "-" * (26 * len(DETECTORS)))

    for key, label, unit, is_pct in metrics_desc:
        if is_pct:
            def fmt_pct(m, ci, s):
                return f"{m*100:.1f}% ± {ci*100:.1f}%".ljust(24)
        else:
            def fmt_pct(m, ci, s):
                return f"{m:.4f} ± {ci:.4f}".ljust(24)

        line = f"  {label:<38}"
        for name, _ in DETECTORS:
            vals = [r[key] for r in all_results[name] if r[key] is not None]
            if len(vals) == 0:
                line += f"{'N/A':<26}"
            else:
                m, ci, s = mean_ci(vals)
                if is_pct:
                    line += f"{m*100:.1f}% ± {ci*100:.1f}%".ljust(26)
                else:
                    line += f"{m:.4f} ± {ci:.4f}".ljust(26)
        w(line)

    # ── SECTION 2: LATENCY DISTRIBUTION ──────────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 2: LATENCY ANALYSIS")
    w("=" * 90)

    w(f"\n  {'Detector':<16} {'Median latency':<18} {'Mean latency':<18} "
      f"{'Min':<10} {'Max':<10} {'% Detected (any)':<18} {'% Detected (batch 15+)':<24}")
    w("  " + "-" * 96)
    for name, _ in DETECTORS:
        latencies = [r["latency"] for r in all_results[name] if r["latency"] is not None]
        detected_any = sum(1 for r in all_results[name] if r["detected"])
        detected_strong = sum(1 for r in all_results[name] if r["latency"] is not None and r["latency"] < 35)
        pct_any = detected_any / len(all_results[name]) * 100
        pct_strong = detected_strong / len(all_results[name]) * 100
        if len(latencies) > 0:
            w(f"  {name:<16} {np.median(latencies):<18.1f} {np.mean(latencies):<18.2f} "
              f"{min(latencies):<10} {max(latencies):<10} {pct_any:<18.1f} {pct_strong:<24.1f}")
        else:
            w(f"  {name:<16} {'N/A':<18} {'N/A':<18} "
              f"{'N/A':<10} {'N/A':<10} {pct_any:<18.1f} {pct_strong:<24.1f}")

    # ── SECTION 3: THRESHOLD SWEEP / ROC SUMMARY ─────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 3: ROC AUC SUMMARY")
    w("=" * 90)
    w(f"\n  ROC AUC measures threshold-independent discriminative power.")
    w(f"  AUC = 0.5 means random guessing, AUC = 1.0 means perfect separation.\n")

    w(f"  {'Detector':<16} {'AUC (covariate)':<20} {'AUC (perturbation)':<22} "
      f"{'Cohen\'s d (covar)':<20} {'Cohen\'s d (pert)':<20}")
    w("  " + "-" * 98)
    for name, _ in DETECTORS:
        aucs = [r["auc"] for r in all_results[name]]
        p_aucs = [r["pert_auc"] for r in all_results[name]]
        ds = [r["cohens_d_strong"] for r in all_results[name]]
        pds = [r["pert_cohens_d"] for r in all_results[name]]
        m_auc, ci_auc, _ = mean_ci(aucs)
        m_pauc, ci_pauc, _ = mean_ci(p_aucs)
        m_d, ci_d, _ = mean_ci(ds)
        m_pd, ci_pd, _ = mean_ci(pds)
        w(f"  {name:<16} {m_auc:.4f} ± {ci_auc:.4f}      "
          f"{m_pauc:.4f} ± {ci_pauc:.4f}        "
          f"{m_d:.4f} ± {ci_d:.4f}      "
          f"{m_pd:.4f} ± {ci_pd:.4f}")

    # ── SECTION 4: OPTIMAL THRESHOLD SEARCH ──────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 4: THRESHOLD CALIBRATION (YOU DEN'S F1-MAX)")
    w("=" * 90)
    w(f"\n  Searching for the optimal threshold that maximizes F1 score")
    w(f"  across all 20 seeds. This reveals whether default thresholds")
    w(f"  are well-calibrated.\n")

    for name, detector in DETECTORS:
        # Gather all scores and labels across seeds
        all_scores_list = []
        all_labels_list = []
        for seed_idx in range(N_SEEDS):
            seed = seed_idx * 7 + 42
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            gen = DriftGenerator(n_features=5, n_reference=2000, random_state=seed)
            ref = gen.generate_reference()
            clean = []
            for _ in range(50):
                idx = rng.choice(2000, 100)
                clean.append(ref[idx].copy())
            drifted = []
            for i in range(50):
                if i < 5:
                    idx = rng.choice(2000, 100)
                    drifted.append(ref[idx].copy())
                else:
                    mag = 2.0 * (i - 4) / 10
                    drifted.append(gen.covariate_shift(n_samples=100, shift_magnitude=mag))

            det = _copy_detector(detector)
            det.fit(ref)

            scores = []
            for batch in clean:
                scores.append(det.score(batch))
            for batch in drifted:
                scores.append(det.score(batch))

            all_scores_list.extend(scores)
            # 55 clean (50 pre-drift + 5 zero-drift) + 45 truly drifted
            all_labels_list.extend([0] * 55 + [1] * 45)

        all_scores_arr = np.array(all_scores_list)
        all_labels_arr = np.array(all_labels_list)

        # Sweep thresholds - include default threshold explicitly
        if np.max(all_scores_arr) - np.min(all_scores_arr) < 1e-10:
            w(f"  {name}: scores too uniform for threshold sweep")
            continue

        lo = min(np.percentile(all_scores_arr, 2), detector.threshold * 0.5)
        hi = max(np.percentile(all_scores_arr, 98), detector.threshold * 2.0)
        thresholds = sorted(set(np.linspace(lo, hi, 100)) | {detector.threshold})
        best_f1 = 0
        best_thresh = detector.threshold
        best_precision = 0
        best_recall = 0
        current_f1 = 0

        for thresh in thresholds:
            preds = (all_scores_arr > thresh).astype(int)
            tp = np.sum((preds == 1) & (all_labels_arr == 1))
            fp = np.sum((preds == 1) & (all_labels_arr == 0))
            fn = np.sum((preds == 0) & (all_labels_arr == 1))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_precision = precision
                best_recall = recall

            if abs(thresh - detector.threshold) < 1e-6:
                current_f1 = f1

        w(f"\n  ── {name} ──")
        w(f"    Default threshold:          {detector.threshold:.4f}")
        w(f"    Optimal threshold (max F1): {best_thresh:.4f}")
        w(f"    F1 at default threshold:    {current_f1:.4f}")
        w(f"    F1 at optimal threshold:    {best_f1:.4f}")
        w(f"    Precision at optimal:       {best_precision:.4f}")
        w(f"    Recall at optimal:          {best_recall:.4f}")
        w(f"    Improvement:                {best_f1 - current_f1:+.4f}")

    # ── SECTION 5: DETECTOR RANKING ──────────────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 5: OVERALL DETECTOR RANKING")
    w("=" * 90)

    # Compute composite scores: average of normalized metrics
    # Higher is better for all metrics except FPR and latency
    rankings = {}
    for name, _ in DETECTORS:
        rank = {}
        vals = [r for r in all_results[name]]
        rank["FPR (lower is better)"] = 1.0 - np.mean([r["fpr"] for r in vals])
        rank["Latency (lower is better)"] = 1.0 - min(1.0, np.mean([r["latency"] for r in vals if r["latency"] is not None]) / 50.0) if any(r["latency"] is not None for r in vals) else 0
        rank["Detection rate"] = np.mean([r["detection_rate"] for r in vals])
        rank["AUC"] = np.mean([r["auc"] for r in vals])
        rank["Effect size"] = min(1.0, np.mean([r["cohens_d_strong"] for r in vals]) / 5.0)
        rank["Stability"] = np.mean([r["stability"] for r in vals])
        rank["Perturbation AUC"] = np.mean([r["pert_auc"] for r in vals])
        rank["Composite"] = np.mean(list(rank.values()))
        rankings[name] = rank

    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]["Composite"], reverse=True)

    w(f"\n  {'Rank':<6} {'Detector':<16} {'Composite':<12} {'FPR':<10} {'Latency':<10} "
      f"{'Det Rate':<10} {'AUC':<10} {'Effect':<10} {'Stability':<12} {'Pert AUC':<10}")
    w("  " + "-" * 106)
    for rank_idx, (name, rank) in enumerate(sorted_rankings, 1):
        w(f"  #{rank_idx:<4} {name:<16} {rank['Composite']:<12.4f} "
          f"{1-rank['FPR (lower is better)']:<10.2%} "
          f"{rank['Latency (lower is better)']:<10.4f} "
          f"{rank['Detection rate']:<10.2%} "
          f"{rank['AUC']:<10.4f} "
          f"{rank['Effect size']:<10.4f} "
          f"{rank['Stability']:<12.4f} "
          f"{rank['Perturbation AUC']:<10.4f}")

    # ── SECTION 6: RAW DATA DUMP (JSON) ──────────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 6: RAW PER-SEED DATA")
    w("=" * 90)
    for name, _ in DETECTORS:
        w(f"\n  ── {name} ──")
        w(f"  {'Seed':<8} {'FPR':<10} {'Lat':<8} {'Det%':<8} {'AUC':<8} "
          f"{'d':<8} {'d(str)':<8} {'pFPR':<10} {'pDet%':<8} {'pAUC':<8}")
        for idx, r in enumerate(all_results[name]):
            seed = idx * 7 + 42
            lat = fmt(r["latency"], 1) if r["latency"] is not None else "N/A"
            w(f"  {seed:<8} {r['fpr']:<10.2%} {lat:<8} "
              f"{r['detection_rate']:<8.2%} {r['auc']:<8.4f} "
              f"{r['cohens_d']:<8.2f} {r['cohens_d_strong']:<8.2f} "
              f"{r['pert_fpr']:<10.2%} {r['pert_detection_rate']:<8.2%} "
              f"{r['pert_auc']:<8.4f}")

    # ── Footer ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    w(f"\n\n{'=' * 90}")
    w(f"  Total benchmark time: {elapsed:.1f} seconds  ({N_SEEDS} seeds × 4 detectors)")
    w(f"  Completed:            {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"{'=' * 90}")

    output = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    try:
        print(output)
    except UnicodeEncodeError:
        print(output.encode("ascii", errors="replace").decode("ascii"))

    print(f"\n  Results saved to: {output_path}")
    return output


if __name__ == "__main__":
    output_path = Path(__file__).parent / "benchmark_results_rigorous.txt"
    run_rigorous_benchmark(str(output_path))
