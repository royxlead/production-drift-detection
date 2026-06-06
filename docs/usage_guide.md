# Usage Guide

## Table of Contents

1. [Full Monitoring Pipeline](#1-full-monitoring-pipeline)
2. [Custom Detector Configuration](#2-custom-detector-configuration)
3. [Alerting Strategies](#3-alerting-strategies)
4. [Synthetic Drift Experiments](#4-synthetic-drift-experiments)
5. [Confidence-Drift Correlation Analysis](#5-confidence-drift-correlation-analysis)
6. [Integration Examples](#6-integration-examples)
7. [Benchmarking Detectors](#7-benchmarking-detectors)
8. [Exporting Results](#8-exporting-results)

---

## 1. Full Monitoring Pipeline

Build a complete production monitoring pipeline:

```python
import numpy as np
from driftwatch.monitors.stream_monitor import StreamMonitor
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.detectors.adwin import ADWINDetector

# 1. Configure detectors
detectors = {
    "kl": KLDivergenceDetector(threshold=0.1),
    "psi": PSIDetector(threshold=0.1, per_feature=True),
    "mmd": MMDDetector(threshold=0.05),
    "adwin": ADWINDetector(threshold=0.1, delta=0.05),
}

# 2. Create monitors
stream = StreamMonitor(detectors=detectors)
confidence = ConfidenceMonitor()
correlation = ConfidenceDriftCorrelation(max_lag=5)

# 3. Fit on reference data
reference = np.random.normal(0, 1, (1000, 5))
stream.fit(reference)

# 4. Process streaming batches
for batch_idx in range(20):
    # Ingest batch
    batch = np.random.normal(batch_idx * 0.15, 1, (100, 5))
    result = stream.process_batch(batch)

    # Simulate model predictions
    probs = np.random.dirichlet(
        [1, 9 - min(batch_idx * 0.3, 5)], size=100
    )
    conf_update = confidence.update(probs)

    # Record for correlation analysis
    correlation.add_observation(
        confidence=conf_update["mean_confidence"],
        drift_scores=result["scores"],
        entropy=conf_update["mean_entropy"],
        margin=conf_update["mean_margin"],
    )

    # Check status
    if result["drift_detected"]:
        print(f"Batch {batch_idx}: DRIFT - {result['status'].upper()}")
    else:
        print(f"Batch {batch_idx}: OK")

# 5. Generate reports
print(stream.summary())
print(confidence.summary())
print(correlation.summary())
```

---

## 2. Custom Detector Configuration

### Tuning KL Divergence

```python
from driftwatch.detectors.kl import KLDivergenceDetector

# For categorical data with smoothing
kl = KLDivergenceDetector(
    threshold=0.05,       # Lower threshold = more sensitive
    is_categorical=True,  # Treat as discrete categories
    smoothing=1e-4,       # Laplace smoothing for zero counts
)
```

### Tuning PSI

```python
from driftwatch.detectors.psi import PSIDetector

# Per-feature analysis
psi = PSIDetector(
    threshold=0.1,
    n_bins=20,             # More bins = more granular
    binning_strategy="uniform",  # Equal-width bins vs quantile
    per_feature=True,      # Track each feature individually
)
```

### Tuning MMD

```python
from driftwatch.detectors.mmd import MMDDetector

mmd = MMDDetector(
    threshold=0.02,        # Strict threshold
    bandwidth=1.0,         # Fixed bandwidth (None = auto median heuristic)
    subsample=500,         # Use up to 500 samples for efficiency
)
```

### Tuning ADWIN

```python
from driftwatch.detectors.adwin import ADWINDetector

adwin = ADWINDetector(
    threshold=0.05,
    delta=0.01,            # Lower delta = more sensitive to small changes
    min_window_size=20,    # Longer warm-up before detection
)
```

---

## 3. Alerting Strategies

### Basic Threshold Alerts

```python
from driftwatch.alerts.rules import AlertEngine, ThresholdRule

# Standard thresholds
engine = AlertEngine(rules=[
    ThresholdRule(
        warning_threshold=0.1,
        critical_threshold=0.25,
    ),
])

alerts = engine.evaluate("psi", score=0.32, threshold=0.1)
for alert in alerts:
    print(f"[{alert.severity.name}] {alert.explanation}")
```

### Rolling Window Consecutive Alerts

```python
from driftwatch.alerts.rules import RollingWindowRule

# Alert only if 3 consecutive batches exceed threshold
rolling = RollingWindowRule(
    threshold=0.1,
    window_size=3,
)

engine = AlertEngine(rules=[ThresholdRule(), rolling])
alerts = engine.evaluate("mmd", score=0.12, threshold=0.05)
```

### Combined Alert Engine

```python
from driftwatch.alerts.rules import AlertEngine, ThresholdRule, RollingWindowRule

engine = AlertEngine(rules=[
    ThresholdRule(
        warning_threshold=0.1,
        critical_threshold=0.25,
    ),
    RollingWindowRule(
        threshold=0.08,
        window_size=5,
    ),
])
```

---

## 4. Synthetic Drift Experiments

### Comparing Drift Types

```python
from driftwatch.data.synthetic_drift import DriftGenerator

generator = DriftGenerator(n_features=5, n_reference=2000, random_state=42)
reference = generator.generate_reference()

# Test different drift types
drifts = {
    "covariate": generator.covariate_shift(500, shift_magnitude=2.0),
    "prior": generator.prior_shift(500, shift_magnitude=1.5),
    "gradual": generator.gradual_drift(10, 50, 0.0, 3.0),
    "sudden": generator.sudden_drift(500, shift_magnitude=3.0),
    "missingness": generator.missingness_drift(reference[:500], missing_rate=0.3),
    "perturbation": generator.feature_perturbation_drift(reference[:500], noise_std=2.0),
    "gaussian": generator.gaussian_noise_drift(reference[:500], noise_std=1.5),
    "corruption": generator.feature_corruption_drift(reference[:500]),
}

for name, data in drifts.items():
    print(f"{name}: shape={data.shape}, nans={np.isnan(data).sum()}")
```

### Reproducible Experiments

```python
# Same random_state = same drift patterns
gen1 = DriftGenerator(n_features=5, random_state=42)
gen2 = DriftGenerator(n_features=5, random_state=42)

ref1 = gen1.generate_reference()
ref2 = gen2.generate_reference()
assert np.allclose(ref1, ref2)  # Identical

shift1 = gen1.covariate_shift(100, shift_magnitude=1.0)
shift2 = gen2.covariate_shift(100, shift_magnitude=1.0)
assert np.allclose(shift1, shift2)  # Identical
```

---

## 5. Confidence-Drift Correlation Analysis

```python
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation

corr = ConfidenceDriftCorrelation(max_lag=5)

# Simulate scenario: confidence drops before drift increases
import numpy as np

for i in range(20):
    if i < 8:
        confidence = 0.92
        drift = 0.02
    elif i < 12:
        confidence = 0.85 - (i - 8) * 0.02  # Confidence drops first
        drift = 0.03
    else:
        confidence = 0.77 - (i - 12) * 0.015
        drift = 0.05 + (i - 12) * 0.04  # Drift accelerates later

    corr.add_observation(
        confidence=confidence,
        drift_scores={"psi": drift},
        entropy=-confidence * np.log(confidence) - (1 - confidence) * np.log(1 - confidence),
        margin=abs(2 * confidence - 1),
    )

# Analyze
cc = corr.compute_cross_correlation()
summary = corr.summary()

print(f"Early warning score: {summary['early_warning_score']:.1f}/100")
print(f"Confidence leads drift: {summary.get('confidence_is_leading_indicator')}")
print(f"Max cross-correlation: {cc.get('max_cross_correlation', 'N/A')}")

# Get visualization data
viz = corr.get_visualization_data()
```

---

## 6. Integration Examples

### Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from driftwatch.integrations.sklearn_adapter import SklearnAdapter
from driftwatch.data.loaders import DataLoader

loader = DataLoader(random_state=42)
data = loader.load_synthetic_classification()

model = RandomForestClassifier()
model.fit(data["X_train"], data["y_train"])

adapter = SklearnAdapter(model)
result = adapter.predict(data["X_test"], data["y_test"])
print(f"Drift detected: {result['drift']['drift_detected']}")
print(f"Confidence: {result['confidence']['mean_confidence']:.3f}")
```

### PyTorch

```python
from driftwatch.integrations.pytorch_adapter import PyTorchAdapter

model = PyTorchAdapter.create_feedforward_classifier(
    input_dim=10, hidden_dim=64, num_classes=2
)
adapter = PyTorchAdapter(model)
```

### HuggingFace

```python
from driftwatch.integrations.hf_adapter import HFAdapter

adapter = HFAdapter()
result = adapter.predict_text([
    "This movie was fantastic!",
    "Terrible experience, would not recommend.",
])
print(f"Drift detected: {result['drift']['drift_detected']}")
```

---

## 7. Benchmarking Detectors

```python
from driftwatch.evaluation.benchmarks import BenchmarkFramework
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector

framework = BenchmarkFramework(detectors=[
    KLDivergenceDetector(threshold=0.1),
    PSIDetector(threshold=0.1),
    MMDDetector(threshold=0.05),
])

# Standard comparison
results = framework.run_benchmark(drift_magnitude=2.0)
print(results)

# Sensitivity analysis
sensitivity = framework.run_sensitivity_analysis(
    magnitudes=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
)
```

Or run the full benchmark suite:

```bash
python benchmarks.py
```

---

## 8. Exporting Results

### From StreamMonitor

```python
# JSON string
json_str = stream.export_results(format="json")

# Pandas DataFrame
df = stream.export_results(format="csv")

# Write directly to file
stream.export_results(format="csv", filepath="drift_scores.csv")
stream.export_results(format="json", filepath="drift_scores.json")
```

### From Demo Script

```bash
# Export summary to JSON
python demo.py --export results.json

# Export drift scores to CSV
python demo.py --export-csv scores.csv

# Export both
python demo.py --export results.json --export-csv scores.csv
```

### Alert History

```python
alerts = stream.get_alerts(severity=Severity.WARNING)
for alert in alerts:
    print(f"[{alert.severity.name}] {alert.timestamp}: "
          f"{alert.detector_name} score={alert.score:.4f}")
```
