# API Reference

## Detectors

### BaseDetector

```python
from driftwatch.detectors.base import BaseDetector
```

Abstract base class for all drift detectors.

**Methods:**
- `fit(reference_data)` — Store reference distribution
- `update(batch)` — Update internal state with new batch
- `score(batch)` — Compute drift score (returns float)
- `detect(batch)` — Score + threshold check (returns dict with `score`, `drift_detected`, `threshold`)
- `summary()` — Current state dictionary
- `reset()` — Reset detector to initial state

---

### KLDivergenceDetector

```python
from driftwatch.detectors.kl import KLDivergenceDetector

detector = KLDivergenceDetector(
    threshold=0.1,
    is_categorical=False,
    n_bins=20,
    epsilon=1e-10,
    smoothing=1e-6,
)
```

Kullback-Leibler Divergence between reference and production distributions.

- **`is_categorical`**: If `True`, treats data as categorical. If `False`, discretizes via binning.
- **`n_bins`**: Number of bins for discretizing continuous data.
- **`epsilon`**: Numerical stability offset.
- **`smoothing`**: Laplace-style smoothing for zero-probability bins.

---

### PSIDetector

```python
from driftwatch.detectors.psi import PSIDetector

detector = PSIDetector(
    threshold=0.1,
    n_bins=10,
    binning_strategy="quantile",
    per_feature=True,
    critical_threshold=0.2,
)
```

Population Stability Index — measures distribution stability via binned proportions.

- **`n_bins`**: Number of bins per feature.
- **`binning_strategy`**: `"quantile"` (equal frequency) or `"uniform"` (equal width).
- **`per_feature`**: If `True`, returns per-feature drift scores.
- **`critical_threshold`**: Threshold for critical-level alerts.

**Standard interpretation:**
- PSI < 0.1: No significant change
- PSI 0.1–0.25: Moderate change
- PSI > 0.25: Significant change

---

### MMDDetector

```python
from driftwatch.detectors.mmd import MMDDetector

detector = MMDDetector(
    threshold=0.05,
    bandwidth=None,   # auto via median heuristic
    subsample=200,
)
```

Maximum Mean Discrepancy — kernel-based two-sample test using RBF (Gaussian) kernels.

- **`bandwidth`**: RBF kernel bandwidth. `None` uses the median heuristic for automatic selection.
- **`subsample`**: Maximum number of samples to use per batch for efficient computation.

---

### ADWINDetector

```python
from driftwatch.detectors.adwin import ADWINDetector

detector = ADWINDetector(
    threshold=0.1,
    delta=0.05,
    min_window_size=10,
)
```

Adaptive Windowing — streaming drift detection with dynamic window resizing.

- **`delta`**: Confidence parameter for the Hoeffding bound (lower = more sensitive).
- **`min_window_size`**: Minimum window size before detection begins.

---

## Monitors

### StreamMonitor

```python
from driftwatch.monitors.stream_monitor import StreamMonitor

monitor = StreamMonitor(
    detectors=None,       # Auto-creates KL, PSI, MMD by default
    alert_engine=None,    # Auto-creates with threshold rules
    window_size=10,
    name="StreamMonitor",
)
```

Orchestrates drift detection across multiple detectors for streaming data.

**Methods:**
- `fit(reference_data)` — Fit all detectors on reference distribution
- `process_batch(batch)` — Process a single data batch through all detectors
- `get_history(detector_name=None)` — Get drift score history
- `get_alerts(severity=None, limit=100)` — Get triggered alerts
- `summary()` — Comprehensive monitoring state summary
- `export_results(format="json", filepath=None)` — Export results to JSON or CSV
- `reset()` — Reset to initial state

**`process_batch` returns:**
```python
{
    "scores": {"kl": 0.15, "psi": 0.32, "mmd": 0.08, "adwin": 0.04},
    "alerts": [...],      # Triggered alert dicts
    "drift_detected": True,
    "batch": 7,
    "status": "warning",  # "healthy", "watch", "warning", or "critical"
}
```

---

### ConfidenceMonitor

```python
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor

monitor = ConfidenceMonitor(
    window_size=10,
    name="ConfidenceMonitor",
    confidence_threshold=0.5,
    entropy_threshold=0.5,
)
```

Tracks prediction confidence, entropy, and margin over time to detect early degradation signals.

**Methods:**
- `update(probabilities, ground_truth=None)` — Update with prediction probabilities
- `get_trends()` — Analyze confidence trends (direction, magnitude)
- `get_uncertainty_metrics()` — Summary of uncertainty for most recent batch
- `degradation_detected()` — Boolean check + reason string
- `get_calibration_summary()` — Calibration quality estimate
- `summary()` — Comprehensive summary
- `reset()` — Reset to initial state

**Tracked metrics:**
- **Confidence**: Maximum predicted probability per sample
- **Entropy**: Predictive distribution entropy
- **Margin**: Difference between top two predicted probabilities
- **Accuracy**: When ground truth is provided

---

## Alerts

### Alert

```python
from driftwatch.alerts.schemas import Alert, Severity

alert = Alert(
    detector_name="psi",
    score=1.64,
    threshold=0.2,
    timestamp=...,
    severity=Severity.CRITICAL,
    explanation="Critical drift: score 1.64 exceeds critical threshold 0.20",
)
```

### Alert Rules

```python
from driftwatch.alerts.rules import ThresholdRule, RollingWindowRule, AlertEngine

# Alert when score exceeds threshold
rule = ThresholdRule(
    warning_threshold=0.1,
    critical_threshold=0.2,
)

# Alert when score exceeds threshold for N consecutive batches
rolling = RollingWindowRule(
    threshold=0.1,
    window_size=3,
)

engine = AlertEngine(rules=[rule, rolling])
alerts = engine.evaluate(detector_name="psi", score=1.64, threshold=0.1)
```

### Severity Levels

| Level | Value | Description |
|-------|-------|-------------|
| `HEALTHY` | 0 | No drift detected |
| `WATCH` | 1 | Slight drift, monitor closely |
| `WARNING` | 2 | Moderate drift, investigate |
| `CRITICAL` | 3 | Severe drift, take action |

---

## Data

### DriftGenerator

```python
from driftwatch.data.synthetic_drift import DriftGenerator

generator = DriftGenerator(
    n_features=5,
    n_reference=1000,
    random_state=42,
)
```

Generates synthetic data with controlled drift for reproducible experiments.

**Drift types:**
- `generate_reference()` — Clean reference distribution
- `covariate_shift(n_samples, shift_magnitude)` — Shift feature means
- `prior_shift(n_samples, shift_magnitude)` — Change class balance
- `gradual_drift(n_steps, n_per_step, start_magnitude, end_magnitude)` — Slowly increasing shift
- `sudden_drift(n_samples, shift_magnitude, abrupt_fraction)` — Abrupt point shift
- `missingness_drift(data, missing_rate)` — Inject NaN values
- `feature_perturbation_drift(data, noise_std, n_corrupt_features)` — Add noise to subset
- `gaussian_noise_drift(data, noise_std)` — Add noise to all features
- `feature_corruption_drift(data, corrupt_value, n_corrupt_features)` — Set features to constants

### DataLoader

```python
from driftwatch.data.loaders import DataLoader

loader = DataLoader(random_state=42)
data = loader.load_synthetic_classification(n_samples=1000, n_features=10)
batches = loader.stream_batches(data["X_train"], batch_size=100, with_drift=True)
```

---

## Correlation (Research Module)

### ConfidenceDriftCorrelation

```python
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation

corr = ConfidenceDriftCorrelation(max_lag=5)

# Track observations over time
corr.add_observation(
    confidence=0.92,
    drift_scores={"psi": 0.02, "kl": 0.01},
    entropy=0.25,
    margin=0.85,
)

# Analyze lead-lag relationships
cross_corr = corr.compute_cross_correlation()

# Get summary
summary = corr.summary()
# Returns early_warning_score, n_early_warnings, confidence_is_leading_indicator, etc.
```

---

## Evaluation

```python
from driftwatch.evaluation.metrics import (
    compute_detection_latency,
    compute_false_positive_rate,
    compute_detection_stability,
    compute_sensitivity_to_drift,
    evaluate_detector,
)

from driftwatch.evaluation.benchmarks import (
    BenchmarkFramework,
    benchmark_detector,
    compare_detectors,
)
```

**Metrics:**
- **Detection latency**: How quickly drift is detected after introduction
- **False positive rate**: Alerts triggered on clean (non-drifted) data
- **Detection stability**: Coefficient of variation across scores (lower = more stable)
- **Sensitivity**: Score change per unit of drift magnitude
