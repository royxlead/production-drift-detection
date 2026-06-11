#!/usr/bin/env python3
"""
DriftWatch — Empirical Validation: Confidence-Drift Correlation

This script empirically tests the core research hypothesis:

    H1: Under gradual covariate shift, changes in model confidence
        precede changes in drift scores by k time steps.

    H2: The lead-lag relationship is asymmetric — confidence is more
        likely to lead drift than vice versa.

Methodology:
  - Trains real classifiers (sklearn LogisticRegression, PyTorch NN)
  - Applies gradual covariate shift to the input features
  - Collects prediction probabilities (→ confidence, entropy, margin)
  - Collects drift scores (KL, PSI, MMD) on the feature distributions
  - Computes cross-correlation between confidence and drift series
  - Reports whether confidence leads drift, and by how many batches

Usage:
    python empirical_validation.py
"""

import sys
import time
import math
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.data.synthetic_drift import DriftGenerator
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor


# ── Detectors for feature drift ──────────────────────────────────
KL_DETECTOR = KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20)
PSI_DETECTOR = PSIDetector(threshold=0.1, n_bins=10)
MMD_DETECTOR = MMDDetector(threshold=0.05, subsample=200)


def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def cohens_d(a, b):
    """Cohen's d effect size between two arrays."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2 + 1e-10))
    return (m2 - m1) / max(sp, 1e-10)


def compute_cross_correlation_maxlag(conf_series, drift_series, max_lag=10):
    """
    Compute cross-correlation between confidence and drift at various lags.

    Positive lag = confidence leads drift (confidence at t-lag vs drift at t).
    Returns dict with correlations and optimal lag.
    """
    conf = np.array(conf_series)
    drift = np.array(drift_series)
    n = min(len(conf), len(drift))

    if n < max_lag + 3:
        return {"optimal_lag": 0, "confidence_leads": False, "n": n}

    conf = conf[:n]
    drift = drift[:n]

    conf_norm = (conf - np.mean(conf)) / max(np.std(conf), 1e-10)
    drift_norm = (drift - np.mean(drift)) / max(np.std(drift), 1e-10)

    correlations = []
    for lag in range(0, min(max_lag, n - 2)):
        if lag == 0:
            corr = float(np.corrcoef(conf_norm, drift_norm)[0, 1])
            correlations.append({"lag": 0, "correlation": corr, "type": "simultaneous"})
        else:
            # Confidence leads drift
            c_lead = conf_norm[:-lag]
            d_lead = drift_norm[lag:]
            r_lead = float(np.corrcoef(c_lead, d_lead)[0, 1]) if len(c_lead) > 2 else 0.0

            # Drift leads confidence
            c_lag = conf_norm[lag:]
            d_lag = drift_norm[:-lag]
            r_lag = float(np.corrcoef(c_lag, d_lag)[0, 1]) if len(c_lag) > 2 else 0.0

            correlations.append({
                "lag": lag,
                "confidence_leads_drift": r_lead,
                "drift_leads_confidence": r_lag,
            })

    # Determine optimal lag
    if len(correlations) < 2:
        return {"optimal_lag": 0, "confidence_leads": False, "n": n, "correlations": correlations}

    simultaneous = abs(correlations[0].get("correlation", 0))
    lead_corrs = [c.get("confidence_leads_drift", 0) for c in correlations[1:]]
    lag_corrs = [c.get("drift_leads_confidence", 0) for c in correlations[1:]]

    max_lead = max(abs(c) for c in lead_corrs) if lead_corrs else 0
    max_lag_c = max(abs(c) for c in lag_corrs) if lag_corrs else 0

    confidence_leads = max_lead > max_lag_c and max_lead > simultaneous * 1.1
    optimal_lag = 0
    if confidence_leads and lead_corrs:
        optimal_lag = int(np.argmax([abs(c) for c in lead_corrs])) + 1

    return {
        "optimal_lag": optimal_lag,
        "confidence_leads": confidence_leads,
        "max_lead_correlation": max_lead,
        "max_lag_correlation": max_lag_c,
        "simultaneous_correlation": simultaneous,
        "n": n,
        "correlations": correlations,
    }


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: sklearn LogisticRegression on synthetic tabular
# ═══════════════════════════════════════════════════════════════════
def experiment_sklearn_tabular(seed: int = 42) -> dict:
    """Train a LogisticRegression, apply gradual covariate shift,
    collect confidence and drift scores, test H1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n_features = 5
    n_train = 2000
    n_batches = 40
    batch_size = 100
    drift_start = 10
    drift_max = 3.0

    gen = DriftGenerator(n_features=n_features, n_reference=n_train, random_state=seed)
    X_train = gen.generate_reference()

    # Create binary labels based on a decision boundary in feature space
    true_w = rng.normal(0, 1, n_features)
    logits = X_train @ true_w + rng.normal(0, 0.5, n_train)
    y_train = (logits > 0).astype(int)

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)

    # Fit drift detectors on reference (unscaled for feature drift)
    KL_DETECTOR.fit(X_train)
    PSI_DETECTOR.fit(X_train)
    MMD_DETECTOR.fit(X_train)

    # Confidence monitor
    conf_mon = ConfidenceMonitor()
    correlator = ConfidenceDriftCorrelation(max_lag=5)

    # Run batches
    confidence_history = []
    drift_history_kl = []
    drift_history_psi = []
    drift_history_mmd = []
    accuracy_history = []
    batch_magnitudes = []

    for i in range(n_batches):
        if i < drift_start:
            # Clean batch (sample from training distribution)
            idx = rng.choice(n_train, batch_size)
            X_batch = X_train[idx].copy()
            y_batch = y_train[idx]
            mag = 0.0
        else:
            # Gradual drift
            progress = (i - drift_start) / (n_batches - drift_start)
            mag = drift_max * progress
            X_batch = gen.covariate_shift(n_samples=batch_size, shift_magnitude=mag)
            # Generate labels with the same decision boundary
            logits_b = X_batch @ true_w + rng.normal(0, 0.5, batch_size)
            y_batch = (logits_b > 0).astype(int)

        batch_magnitudes.append(mag)

        # Get model predictions
        X_scaled = scaler.transform(X_batch)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
        else:
            preds = model.predict(X_scaled)
            probs = np.zeros((len(X_scaled), 2))
            probs[np.arange(len(X_scaled)), preds] = 0.9
            probs[:, 1 - preds] = 0.1

        # Confidence
        conf_update = conf_mon.update(probs, y_batch)
        mean_conf = conf_update["mean_confidence"]
        confidence_history.append(mean_conf)

        # Accuracy
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_batch))
        accuracy_history.append(acc)

        # Drift scores
        drift_history_kl.append(KL_DETECTOR.score(X_batch))
        drift_history_psi.append(PSI_DETECTOR.score(X_batch))
        drift_history_mmd.append(MMD_DETECTOR.score(X_batch))

        # Correlator
        correlator.add_observation(
            confidence=mean_conf,
            drift_scores={"kl": drift_history_kl[-1], "psi": drift_history_psi[-1], "mmd": drift_history_mmd[-1]},
            entropy=conf_update.get("mean_entropy", 0),
            margin=conf_update.get("mean_margin", 0),
        )

    # Analysis
    pre_drift_conf = np.mean(confidence_history[:drift_start])
    post_drift_conf = np.mean(confidence_history[drift_start:])
    conf_drop_pct = (pre_drift_conf - post_drift_conf) / max(pre_drift_conf, 1e-10) * 100

    # Cross-correlation: confidence vs each drift detector
    xc_kl = compute_cross_correlation_maxlag(confidence_history, drift_history_kl, max_lag=10)
    xc_psi = compute_cross_correlation_maxlag(confidence_history, drift_history_psi, max_lag=10)
    xc_mmd = compute_cross_correlation_maxlag(confidence_history, drift_history_mmd, max_lag=10)

    # Effect sizes
    d_kl = cohens_d(np.array(confidence_history[:drift_start]), np.array(confidence_history[drift_start:]))
    d_drift_kl = cohens_d(np.array(drift_history_kl[:drift_start]), np.array(drift_history_kl[drift_start:]))

    return {
        "model": "LogisticRegression",
        "train_acc": train_acc,
        "n_batches": n_batches,
        "drift_start": drift_start,
        "drift_max": drift_max,
        "pre_drift_mean_conf": float(pre_drift_conf),
        "post_drift_mean_conf": float(post_drift_conf),
        "conf_drop_pct": conf_drop_pct,
        "pre_drift_mean_acc": float(np.mean(accuracy_history[:drift_start])),
        "post_drift_mean_acc": float(np.mean(accuracy_history[drift_start:])),
        "acc_drop_pct": float((np.mean(accuracy_history[:drift_start]) - np.mean(accuracy_history[drift_start:])) / max(np.mean(accuracy_history[:drift_start]), 1e-10) * 100),
        "confidence_leads_drift_kl": xc_kl["confidence_leads"],
        "confidence_leads_drift_psi": xc_psi["confidence_leads"],
        "confidence_leads_drift_mmd": xc_mmd["confidence_leads"],
        "optimal_lag_kl": xc_kl["optimal_lag"],
        "optimal_lag_psi": xc_psi["optimal_lag"],
        "optimal_lag_mmd": xc_mmd["optimal_lag"],
        "max_lead_corr_kl": xc_kl.get("max_lead_correlation", 0),
        "max_lead_corr_psi": xc_psi.get("max_lead_correlation", 0),
        "max_lead_corr_mmd": xc_mmd.get("max_lead_correlation", 0),
        "simultaneous_corr_kl": xc_kl.get("simultaneous_correlation", 0),
        "cohens_d_confidence": float(d_kl),
        "cohens_d_drift_kl": float(d_drift_kl),
        "early_warning_score": correlator.compute_early_warning_score().get("early_warning_score", 0),
        "n_early_warnings": correlator.summary()["n_early_warnings"],
        "confidence_trend": float(confidence_history[-1] - confidence_history[0]),
        "drift_trend_kl": float(drift_history_kl[-1] - drift_history_kl[0]),
        "confidence_history": confidence_history,
        "drift_history_kl": drift_history_kl,
        "drift_history_psi": drift_history_psi,
        "drift_history_mmd": drift_history_mmd,
        "accuracy_history": accuracy_history,
        "batch_magnitudes": batch_magnitudes,
        "xc_kl": xc_kl,
        "xc_psi": xc_psi,
        "xc_mmd": xc_mmd,
        "correlator_summary": correlator.summary(),
        "conf_mon_summary": conf_mon.summary(),
    }


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: PyTorch Neural Network on synthetic tabular
# ═══════════════════════════════════════════════════════════════════
def experiment_pytorch_nn(seed: int = 42) -> dict:
    """Train a PyTorch feedforward NN, apply gradual covariate shift,
    collect confidence and drift scores, test H1."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_features = 5
    n_train = 2000
    n_batches = 40
    batch_size = 100
    drift_start = 10
    drift_max = 3.0

    gen = DriftGenerator(n_features=n_features, n_reference=n_train, random_state=seed)
    X_train = gen.generate_reference()

    true_w = rng.normal(0, 1, n_features)
    logits = X_train @ true_w + rng.normal(0, 0.3, n_train)
    y_train = (logits > 0).astype(int)

    # Build NN
    class SimpleNN(nn.Module):
        def __init__(self, input_dim=5, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cpu")
    model = SimpleNN(n_features, 32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).long()
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(30):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    model.eval()

    with torch.no_grad():
        out = model(X_t)
        train_preds = out.argmax(dim=1).numpy()
        train_acc = float(np.mean(train_preds == y_train))

    # Fit drift detectors
    KL_DETECTOR.fit(X_train)
    PSI_DETECTOR.fit(X_train)
    MMD_DETECTOR.fit(X_train)

    conf_mon = ConfidenceMonitor()
    correlator = ConfidenceDriftCorrelation(max_lag=5)

    confidence_history = []
    drift_history_kl = []
    drift_history_psi = []
    drift_history_mmd = []
    accuracy_history = []
    batch_magnitudes = []

    for i in range(n_batches):
        if i < drift_start:
            idx = rng.choice(n_train, batch_size)
            X_batch = X_train[idx].copy()
            y_batch = y_train[idx]
            mag = 0.0
        else:
            progress = (i - drift_start) / (n_batches - drift_start)
            mag = drift_max * progress
            X_batch = gen.covariate_shift(n_samples=batch_size, shift_magnitude=mag)
            logits_b = X_batch @ true_w + rng.normal(0, 0.3, batch_size)
            y_batch = (logits_b > 0).astype(int)

        batch_magnitudes.append(mag)

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_batch).float()
            logits_out = model(X_tensor)
            probs = torch.softmax(logits_out, dim=1).numpy()

        conf_update = conf_mon.update(probs, y_batch)
        mean_conf = conf_update["mean_confidence"]
        confidence_history.append(mean_conf)

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_batch))
        accuracy_history.append(acc)

        drift_history_kl.append(KL_DETECTOR.score(X_batch))
        drift_history_psi.append(PSI_DETECTOR.score(X_batch))
        drift_history_mmd.append(MMD_DETECTOR.score(X_batch))

        correlator.add_observation(
            confidence=mean_conf,
            drift_scores={"kl": drift_history_kl[-1], "psi": drift_history_psi[-1], "mmd": drift_history_mmd[-1]},
            entropy=conf_update.get("mean_entropy", 0),
            margin=conf_update.get("mean_margin", 0),
        )

    pre_drift_conf = np.mean(confidence_history[:drift_start])
    post_drift_conf = np.mean(confidence_history[drift_start:])
    conf_drop_pct = (pre_drift_conf - post_drift_conf) / max(pre_drift_conf, 1e-10) * 100

    xc_kl = compute_cross_correlation_maxlag(confidence_history, drift_history_kl, max_lag=10)
    xc_psi = compute_cross_correlation_maxlag(confidence_history, drift_history_psi, max_lag=10)
    xc_mmd = compute_cross_correlation_maxlag(confidence_history, drift_history_mmd, max_lag=10)

    d_kl = cohens_d(np.array(confidence_history[:drift_start]), np.array(confidence_history[drift_start:]))

    return {
        "model": "PyTorch NN",
        "train_acc": train_acc,
        "n_batches": n_batches,
        "drift_start": drift_start,
        "drift_max": drift_max,
        "pre_drift_mean_conf": float(pre_drift_conf),
        "post_drift_mean_conf": float(post_drift_conf),
        "conf_drop_pct": conf_drop_pct,
        "pre_drift_mean_acc": float(np.mean(accuracy_history[:drift_start])),
        "post_drift_mean_acc": float(np.mean(accuracy_history[drift_start:])),
        "acc_drop_pct": float((np.mean(accuracy_history[:drift_start]) - np.mean(accuracy_history[drift_start:])) / max(np.mean(accuracy_history[:drift_start]), 1e-10) * 100),
        "confidence_leads_drift_kl": xc_kl["confidence_leads"],
        "confidence_leads_drift_psi": xc_psi["confidence_leads"],
        "confidence_leads_drift_mmd": xc_mmd["confidence_leads"],
        "optimal_lag_kl": xc_kl["optimal_lag"],
        "optimal_lag_psi": xc_psi["optimal_lag"],
        "optimal_lag_mmd": xc_mmd["optimal_lag"],
        "max_lead_corr_kl": xc_kl.get("max_lead_correlation", 0),
        "max_lead_corr_psi": xc_psi.get("max_lead_correlation", 0),
        "max_lead_corr_mmd": xc_mmd.get("max_lead_correlation", 0),
        "simultaneous_corr_kl": xc_kl.get("simultaneous_correlation", 0),
        "cohens_d_confidence": float(d_kl),
        "early_warning_score": correlator.compute_early_warning_score().get("early_warning_score", 0),
        "n_early_warnings": correlator.summary()["n_early_warnings"],
        "confidence_trend": float(confidence_history[-1] - confidence_history[0]),
        "drift_trend_kl": float(drift_history_kl[-1] - drift_history_kl[0]),
        "confidence_history": confidence_history,
        "drift_history_kl": drift_history_kl,
        "drift_history_psi": drift_history_psi,
        "drift_history_mmd": drift_history_mmd,
        "accuracy_history": accuracy_history,
        "batch_magnitudes": batch_magnitudes,
        "xc_kl": xc_kl,
        "xc_psi": xc_psi,
        "xc_mmd": xc_mmd,
        "correlator_summary": correlator.summary(),
    }


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: sklearn on perturbation drift
# ═══════════════════════════════════════════════════════════════════
def experiment_perturbation_drift(seed: int = 42) -> dict:
    """Same as Experiment 1 but with feature perturbation drift."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n_features = 5
    n_train = 2000
    n_batches = 40
    batch_size = 100
    drift_start = 10

    gen = DriftGenerator(n_features=n_features, n_reference=n_train, random_state=seed)
    X_train = gen.generate_reference()

    true_w = rng.normal(0, 1, n_features)
    logits = X_train @ true_w + rng.normal(0, 0.5, n_train)
    y_train = (logits > 0).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)

    KL_DETECTOR.fit(X_train)
    PSI_DETECTOR.fit(X_train)
    MMD_DETECTOR.fit(X_train)

    conf_mon = ConfidenceMonitor()
    correlator = ConfidenceDriftCorrelation(max_lag=5)

    confidence_history = []
    drift_history_kl = []
    drift_history_psi = []
    drift_history_mmd = []
    accuracy_history = []
    batch_noise_stds = []

    for i in range(n_batches):
        if i < drift_start:
            idx = rng.choice(n_train, batch_size)
            X_batch = X_train[idx].copy()
            y_batch = y_train[idx]
            noise_std = 0.0
        else:
            progress = (i - drift_start) / (n_batches - drift_start)
            noise_std = 3.0 * progress
            idx = rng.choice(n_train, batch_size)
            X_batch = X_train[idx].copy()
            X_batch += rng.normal(0, noise_std, X_batch.shape)
            logits_b = X_batch @ true_w + rng.normal(0, 0.5, batch_size)
            y_batch = (logits_b > 0).astype(int)

        batch_noise_stds.append(noise_std)

        X_scaled = scaler.transform(X_batch)
        probs = model.predict_proba(X_scaled)

        conf_update = conf_mon.update(probs, y_batch)
        mean_conf = conf_update["mean_confidence"]
        confidence_history.append(mean_conf)

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_batch))
        accuracy_history.append(acc)

        drift_history_kl.append(KL_DETECTOR.score(X_batch))
        drift_history_psi.append(PSI_DETECTOR.score(X_batch))
        drift_history_mmd.append(MMD_DETECTOR.score(X_batch))

        correlator.add_observation(
            confidence=mean_conf,
            drift_scores={"kl": drift_history_kl[-1], "psi": drift_history_psi[-1], "mmd": drift_history_mmd[-1]},
            entropy=conf_update.get("mean_entropy", 0),
            margin=conf_update.get("mean_margin", 0),
        )

    pre_drift_conf = np.mean(confidence_history[:drift_start])
    post_drift_conf = np.mean(confidence_history[drift_start:])
    conf_drop_pct = (pre_drift_conf - post_drift_conf) / max(pre_drift_conf, 1e-10) * 100

    xc_kl = compute_cross_correlation_maxlag(confidence_history, drift_history_kl, max_lag=10)
    xc_psi = compute_cross_correlation_maxlag(confidence_history, drift_history_psi, max_lag=10)
    xc_mmd = compute_cross_correlation_maxlag(confidence_history, drift_history_mmd, max_lag=10)

    return {
        "model": "LogisticRegression + Perturbation",
        "train_acc": train_acc,
        "n_batches": n_batches,
        "drift_start": drift_start,
        "pre_drift_mean_conf": float(pre_drift_conf),
        "post_drift_mean_conf": float(post_drift_conf),
        "conf_drop_pct": conf_drop_pct,
        "pre_drift_mean_acc": float(np.mean(accuracy_history[:drift_start])),
        "post_drift_mean_acc": float(np.mean(accuracy_history[drift_start:])),
        "acc_drop_pct": float((np.mean(accuracy_history[:drift_start]) - np.mean(accuracy_history[drift_start:])) / max(np.mean(accuracy_history[:drift_start]), 1e-10) * 100),
        "confidence_leads_drift_kl": xc_kl["confidence_leads"],
        "confidence_leads_drift_psi": xc_psi["confidence_leads"],
        "confidence_leads_drift_mmd": xc_mmd["confidence_leads"],
        "optimal_lag_kl": xc_kl["optimal_lag"],
        "optimal_lag_psi": xc_psi["optimal_lag"],
        "optimal_lag_mmd": xc_mmd["optimal_lag"],
        "max_lead_corr_kl": xc_kl.get("max_lead_correlation", 0),
        "max_lead_corr_psi": xc_psi.get("max_lead_correlation", 0),
        "max_lead_corr_mmd": xc_mmd.get("max_lead_correlation", 0),
        "early_warning_score": correlator.compute_early_warning_score().get("early_warning_score", 0),
        "n_early_warnings": correlator.summary()["n_early_warnings"],
        "confidence_trend": float(confidence_history[-1] - confidence_history[0]),
        "drift_trend_kl": float(drift_history_kl[-1] - drift_history_kl[0]),
        "confidence_history": confidence_history,
        "drift_history_kl": drift_history_kl,
        "drift_history_psi": drift_history_psi,
        "drift_history_mmd": drift_history_mmd,
        "accuracy_history": accuracy_history,
        "batch_noise_stds": batch_noise_stds,
        "xc_kl": xc_kl,
        "xc_psi": xc_psi,
        "xc_mmd": xc_mmd,
        "correlator_summary": correlator.summary(),
    }


# ═══════════════════════════════════════════════════════════════════
#  MULTI-SEED AGGREGATION RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_empirical_validation(output_path: str):
    start_time = time.time()
    N_SEEDS = 10

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 90)
    w("  DriftWatch — Empirical Validation: Confidence-Drift Correlation")
    w("=" * 90)
    w(f"  Started:              {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  NumPy:                {np.__version__}")
    w(f"  Random seeds:         {N_SEEDS}")
    w(f"  Batches per trial:    40 (10 clean + 30 gradual drift)")
    w(f"  Batch size:           100")
    w(f"  Drift magnitude:      0 -> 3.0 (linear ramp)")
    w(f"  Drift types:          covariate_shift, perturbation")
    w(f"  Models:               LogisticRegression, PyTorch NN")
    w(f"  Max cross-correlation lag: 10 batches")
    w()
    w("  Hypothesis H1: Confidence degradation precedes drift detection")
    w("  Hypothesis H2: Asymmetric lead-lag (confidence leads drift)")
    w("=" * 90)

    experiments = [
        ("sklearn_covariate", experiment_sklearn_tabular),
        ("pytorch_covariate", experiment_pytorch_nn),
        ("sklearn_perturbation", experiment_perturbation_drift),
    ]

    # Run experiments across seeds
    all_results = {}
    for exp_name, exp_func in experiments:
        all_results[exp_name] = []
        for s in range(N_SEEDS):
            seed = s * 7 + 42
            w(f"\n  Running {exp_name} (seed {seed})...")
            result = exp_func(seed=seed)
            all_results[exp_name].append(result)

    # ── SECTION 1: HYPOTHESIS TEST RESULTS ───────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 1: HYPOTHESIS TEST — Does Confidence Lead Drift?")
    w("=" * 90)

    for exp_name, _ in experiments:
        w(f"\n  ── {exp_name} ──")
        results = all_results[exp_name]

        # Aggregate lead-lag results
        leads_kl = sum(1 for r in results if r["confidence_leads_drift_kl"])
        leads_psi = sum(1 for r in results if r["confidence_leads_drift_psi"])
        leads_mmd = sum(1 for r in results if r["confidence_leads_drift_mmd"])
        total = len(results)

        avg_lag_kl = np.mean([r["optimal_lag_kl"] for r in results])
        avg_lag_psi = np.mean([r["optimal_lag_psi"] for r in results])
        avg_lag_mmd = np.mean([r["optimal_lag_mmd"] for r in results])

        avg_lead_corr_kl = np.mean([r["max_lead_corr_kl"] for r in results])
        avg_lead_corr_psi = np.mean([r["max_lead_corr_psi"] for r in results])
        avg_lead_corr_mmd = np.mean([r["max_lead_corr_mmd"] for r in results])

        avg_ews = np.mean([r["early_warning_score"] for r in results])
        avg_conf_drop = np.mean([r["conf_drop_pct"] for r in results])
        avg_acc_drop = np.mean([r["acc_drop_pct"] for r in results])

        w(f"    H1: Confidence leads drift (KL):     {leads_kl}/{total} seeds  ({leads_kl/total*100:.0f}%)")
        w(f"    H1: Confidence leads drift (PSI):    {leads_psi}/{total} seeds  ({leads_psi/total*100:.0f}%)")
        w(f"    H1: Confidence leads drift (MMD):    {leads_mmd}/{total} seeds  ({leads_mmd/total*100:.0f}%)")
        w(f"    Avg optimal lag (KL):                {avg_lag_kl:.2f} batches")
        w(f"    Avg optimal lag (PSI):               {avg_lag_psi:.2f} batches")
        w(f"    Avg optimal lag (MMD):               {avg_lag_mmd:.2f} batches")
        w(f"    Avg max lead correlation (KL):       {avg_lead_corr_kl:.4f}")
        w(f"    Avg max lead correlation (PSI):      {avg_lead_corr_psi:.4f}")
        w(f"    Avg max lead correlation (MMD):      {avg_lead_corr_mmd:.4f}")
        w(f"    Avg early warning score:             {avg_ews:.2f}/100")
        w(f"    Avg confidence drop:                 {avg_conf_drop:.2f}%")
        w(f"    Avg accuracy drop:                   {avg_acc_drop:.2f}%")

    # ── SECTION 2: DETAILED PER-SEED RESULTS ─────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 2: DETAILED PER-SEED RESULTS")
    w("=" * 90)

    for exp_name, _ in experiments:
        results = all_results[exp_name]
        w(f"\n  ── {exp_name} (model: {results[0]['model']}) ──")
        w(f"  {'Seed':<8} {'Conf Drop%':<12} {'Acc Drop%':<12} {'Leads(KL)':<10} "
          f"{'Leads(PSI)':<10} {'Leads(MMD)':<10} {'Lag(KL)':<8} {'Lag(PSI)':<8} "
          f"{'Lag(MMD)':<8} {'EWS':<6}")
        w("  " + "-" * 100)
        for s, r in enumerate(results):
            seed = s * 7 + 42
            w(f"  {seed:<8} {r['conf_drop_pct']:<12.2f} {r['acc_drop_pct']:<12.2f} "
              f"{fmt(r['confidence_leads_drift_kl']):<10} {fmt(r['confidence_leads_drift_psi']):<10} "
              f"{fmt(r['confidence_leads_drift_mmd']):<10} {r['optimal_lag_kl']:<8} "
              f"{r['optimal_lag_psi']:<8} {r['optimal_lag_mmd']:<8} "
              f"{r['early_warning_score']:<6.0f}")

    # ── SECTION 3: HYPOTHESIS CONCLUSION ──────────────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 3: HYPOTHESIS CONCLUSION")
    w("=" * 90)

    for exp_name, _ in experiments:
        results = all_results[exp_name]
        leads_kl = sum(1 for r in results if r["confidence_leads_drift_kl"])
        leads_psi = sum(1 for r in results if r["confidence_leads_drift_psi"])
        leads_mmd = sum(1 for r in results if r["confidence_leads_drift_mmd"])
        total = len(results)
        avg_ews = np.mean([r["early_warning_score"] for r in results])
        avg_conf_drop = np.mean([r["conf_drop_pct"] for r in results])

        w(f"\n  ── {exp_name} ({results[0]['model']}) ──")
        w(f"    Model:           {results[0]['model']}")
        w(f"    Train accuracy:  {results[0]['train_acc']:.2%}")
        w(f"    Total batches:   {results[0]['n_batches']} ({results[0]['drift_start']} clean + {results[0]['n_batches'] - results[0]['drift_start']} drifted)")
        w(f"    Confidence drop: {avg_conf_drop:.2f}% across drift phase")
        w(f"    H1 support:      Confidence leads drift in")
        w(f"                     {leads_kl}/{total} trials (KL), {leads_psi}/{total} trials (PSI), {leads_mmd}/{total} trials (MMD)")
        w(f"    Avg EWS:         {avg_ews:.2f}/100")
        total_leads = leads_kl + leads_psi + leads_mmd
        ratio = total_leads / (total * 3)
        if ratio > 0.5:
            verdict = "SUPPORTS H1 - Confidence consistently leads drift."
        elif ratio > 0.3:
            verdict = "PARTIAL SUPPORT - Confidence sometimes leads drift."
        else:
            verdict = "DOES NOT SUPPORT H1 - Confidence does not reliably lead drift."
        w(f"    Conclusion:      {ratio:.0%} of detector-seed pairs show confidence leading drift. {verdict}")

    # ── SECTION 4: RAW TIME SERIES (FIRST SEED) ──────────────────
    w("\n\n" + "=" * 90)
    w("  SECTION 4: SAMPLE TIME SERIES (seed=42, sklearn covariate)")
    w("=" * 90)

    sample = all_results["sklearn_covariate"][0]
    w(f"\n  {'Batch':<8} {'Magnitude':<12} {'Confidence':<12} {'Accuracy':<12} {'Drift(KL)':<12} {'Drift(PSI)':<12} {'Drift(MMD)':<12}")
    w("  " + "-" * 68)
    for i in range(len(sample["confidence_history"])):
        w(f"  {i:<8} {sample['batch_magnitudes'][i]:<12.2f} "
          f"{sample['confidence_history'][i]:<12.4f} {sample['accuracy_history'][i]:<12.4f} "
          f"{sample['drift_history_kl'][i]:<12.4f} {sample['drift_history_psi'][i]:<12.4f} "
          f"{sample['drift_history_mmd'][i]:<12.4f}")

    # ── FOOTER ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    w(f"\n\n{'=' * 90}")
    w(f"  Total validation time: {elapsed:.1f} seconds")
    w(f"  Completed:             {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Experiments:           3 (sklearn covariate, pytorch covariate, sklearn perturbation)")
    w(f"  Seeds per experiment:  {N_SEEDS}")
    w(f"  Total trials:          {3 * N_SEEDS}")
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
    output_path = Path(__file__).parent / "empirical_validation_results.txt"
    run_empirical_validation(str(output_path))
