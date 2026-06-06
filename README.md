<div align="center">
  <h1>DriftWatch</h1>
  <p><strong>Real-time data drift detection for deployed machine learning systems</strong></p>
  <p>
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#drift-detection-methods">Methods</a> •
    <a href="#dashboard">Dashboard</a> •
    <a href="RESEARCH_NOTE.md">Research</a>
  </p>
</div>

---

## Overview

DriftWatch is a lightweight, pip-installable Python library for monitoring data drift and model confidence in production ML systems. It detects when real-world data begins diverging from the training distribution — before model performance visibly degrades.

### Key Features

- **4 drift detection methods**: KL Divergence, PSI, MMD, ADWIN-style adaptive windowing
- **Unified detector API** with consistent `fit/score/detect/summary` interface
- **Stream monitoring** for batch ingestion with rolling statistics
- **Confidence monitoring** with entropy, margin, and trend analysis
- **Novel confidence-drift correlation** — detect if confidence changes precede drift detection
- **Alerting system** with severity levels (Healthy → Watch → Warning → Critical)
- **Interactive Streamlit dashboard** with 6 pages of visualizations
- **Synthetic drift generator** for reproducible experiments (8 drift types)
- **Integrations**: Scikit-learn, PyTorch, HuggingFace (DistilBERT)
- **Evaluation framework** with detection latency, FPR, stability, sensitivity metrics
- **Complete test suite** with pytest
- **Type hints, logging, input validation**, and comprehensive documentation

### Target Users

- ML Engineers monitoring deployed models
- MLOps engineers building observability pipelines
- Data scientists tracking model performance
- AI researchers studying distribution shift
- Graduate students in ML/AI programs

---

## Installation

### From source

```bash
git clone https://github.com/yourusername/driftwatch.git
cd driftwatch
pip install -e .
```

### With optional dependencies

```bash
# PyTorch support
pip install -e ".[pytorch]"

# HuggingFace Transformers support
pip install -e ".[transformers]"

# Development (testing, notebooks)
pip install -e ".[dev]"

# All dependencies
pip install -e ".[all]"
```

### Requirements

- Python ≥ 3.9
- NumPy, Pandas, SciPy, Scikit-learn (required)
- Plotly, Streamlit (required for dashboard)
- PyTorch (optional, for PyTorch adapter)
- Transformers (optional, for HuggingFace adapter)

---

## Quick Start

### Basic drift detection

```python
import numpy as np
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector

# Generate reference and production data
reference = np.random.normal(0, 1, (500, 3))
production = np.random.normal(2, 1, (500, 3))

# KL Divergence Detector
kl_detector = KLDivergenceDetector(threshold=0.1)
kl_detector.fit(reference)
result = kl_detector.detect(production)
print(f"KL Divergence: {result['score']:.4f} — Drift: {result['drift_detected']}")

# PSI Detector
psi_detector = PSIDetector(threshold=0.1)
psi_detector.fit(reference)
result = psi_detector.detect(production)
print(f"PSI: {result['score']:.4f} — Drift: {result['drift_detected']}")

# MMD Detector
mmd_detector = MMDDetector(threshold=0.05)
mmd_detector.fit(reference)
result = mmd_detector.detect(production)
print(f"MMD: {result['score']:.4f} — Drift: {result['drift_detected']}")
```

### Streaming monitoring

```python
from driftwatch.monitors.stream_monitor import StreamMonitor

# Create monitor with default detectors
monitor = StreamMonitor()
reference = np.random.normal(0, 1, (500, 3))
monitor.fit(reference)

# Process streaming batches
for batch_idx in range(10):
    batch = np.random.normal(batch_idx * 0.2, 1, (100, 3))
    result = monitor.process_batch(batch)
    print(f"Batch {batch_idx}: Status={result['status']}, Scores={result['scores']}")
```

### Full demo

```bash
python demo.py
python demo.py --dashboard  # Launch Streamlit dashboard
```

---

## Architecture

The library follows a modular structure:

```
driftwatch/
├── __init__.py              # Package init with version
├── detectors/               # Drift detection algorithms
│   ├── base.py              # Abstract base class
│   ├── kl.py                # KL Divergence
│   ├── psi.py               # Population Stability Index
│   ├── mmd.py               # Maximum Mean Discrepancy
│   └── adwin.py             # ADWIN-style adaptive windowing
├── monitors/                # High-level monitors
│   ├── stream_monitor.py    # Batch ingestion and coordination
│   └── confidence_monitor.py # Confidence, entropy, margin tracking
├── alerts/                  # Alerting system
│   ├── schemas.py           # Alert dataclass, Severity enum
│   └── rules.py             # ThresholdRule, RollingWindowRule, AlertEngine
├── correlation/             # Novel research module
│   └── confidence_drift.py  # Confidence-drift correlation analysis
├── data/                    # Data utilities
│   ├── synthetic_drift.py   # 8 drift types for reproducible experiments
│   └── loaders.py           # Dataset loaders for demos
├── integrations/            # Model integration adapters
│   ├── sklearn_adapter.py   # Scikit-learn wrapper
│   ├── pytorch_adapter.py   # PyTorch wrapper
│   └── hf_adapter.py        # HuggingFace DistilBERT wrapper
├── evaluation/              # Evaluation framework
│   ├── metrics.py           # Latency, FPR, stability, sensitivity
│   └── benchmarks.py        # Benchmark framework + external tool compat
├── dashboard/               # Streamlit visualization
│   ├── app.py               # Multi-page dashboard
│   └── visuals.py           # Plotly visualization helpers
└── utils/                   # Shared utilities
    ├── validation.py        # Input validation
    ├── logging.py           # Logging configuration
    └── stats.py             # Statistical helpers
```

### Unified Detector API

Every detector exposes the same interface:

```python
fit(reference_data)    # Store reference distribution
update(batch)          # Update internal state
score(batch)           # Compute drift score
detect(batch)          # Score + threshold check
summary()              # Current state dictionary
```

---

## Drift Detection Methods

### 1. KL Divergence

Measures the Kullback-Leibler divergence between reference and actual probability distributions. Best for categorical features and probability outputs.

- Smoothing support with Laplace-style epsilon
- Numerical stability guarantees (always non-negative)
- Threshold-based detection

### 2. Population Stability Index (PSI)

Measures the stability of feature distributions over time using binned proportions.

- Configurable binning (quantile or uniform)
- Per-feature drift reporting
- Aggregate drift scoring
- Standard convention: PSI < 0.1 (no change), 0.1-0.25 (moderate), > 0.25 (significant)

### 3. Maximum Mean Discrepancy (MMD)

A kernel-based two-sample test using RBF (Gaussian) kernels. Supports multivariate distributions.

- Median heuristic for automatic bandwidth selection
- Configurable subsample for large datasets
- Biased MMD² estimator for computational efficiency

### 4. ADWIN-Style Adaptive Windowing

A simplified but statistically credible implementation of the ADWIN algorithm for streaming drift detection.

- Adaptive window resizing based on detected changes
- Hoeffding-bound-based change detection
- Online drift detection without storing all historical data

---

## Confidence Monitoring

The `ConfidenceMonitor` tracks model prediction confidence over time, serving as an early warning system before accuracy degradation.

**Tracked metrics:**
- **Confidence**: Maximum predicted probability per sample
- **Entropy**: Uncertainty in the predictive distribution
- **Margin**: Difference between top two predicted probabilities
- **Trend analysis**: Direction and magnitude of metric changes
- **Degradation detection**: Automatic flagging of confidence drops

**Calibration summary:**
- Confidence distribution histograms
- Over/underconfident prediction ratios
- ECE (Expected Calibration Error) approximation

---

## Novel Research Module: Confidence-Drift Correlation

The **confidence-drift correlation** module is DriftWatch's differentiating feature, addressing a key research question:

> *Can changes in model confidence serve as earlier warning signals than drift scores or accuracy degradation?*

### Approach

1. **Tracks confidence changes over time** via the ConfidenceMonitor
2. **Tracks drift scores over time** via the StreamMonitor
3. **Correlates the two series** at different time lags
4. **Estimates lead-lag relationships** (does confidence lead drift?)
5. **Generates early warning indicators** when confidence degrades before drift

### Components

- **Cross-correlation analysis**: Computes correlations at lags from 0 to max_lag
- **Lead-lag estimation**: Determines if one series leads the other
- **Early warning score**: Composite metric (0-100) combining:
  - Confidence degradation rate
  - Drift acceleration
  - Leading indicator status
- **Visualization data**: Formatted for correlation plots and time series

### Research Context

This module connects to established research in:
- **Self-diagnosing neural models** (Leibig et al., 2017)
- **Bayesian uncertainty** (Gal & Ghahramani, 2016)
- **Out-of-distribution detection** (Hendrycks & Gimpel, 2016)
- **Conformal prediction** (Vovk et al., 2005)
- **Calibration-aware monitoring** (Guo et al., 2017)

See [RESEARCH_NOTE.md](RESEARCH_NOTE.md) for the full theoretical background.

---

## Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run src/driftwatch/dashboard/app.py
```

Or after running the demo:

```bash
python demo.py --dashboard
```

### Pages

1. **Overview**: Current status, active alerts, recent drift scores
2. **Drift Monitoring**: PSI, MMD, KL, ADWIN trends over time
3. **Feature-Level Analysis**: Per-feature drift heatmap and ranking
4. **Confidence Monitoring**: Confidence, entropy, margin history
5. **Confidence-Drift Correlation**: Lead-lag plots, cross-correlation, early warnings
6. **Alerts**: Filterable alert log with severity indicators

---

## Examples

### Scikit-learn Integration

```python
from sklearn.ensemble import RandomForestClassifier
from driftwatch.integrations.sklearn_adapter import SklearnAdapter
from driftwatch.data.loaders import DataLoader

# Load data
loader = DataLoader(random_state=42)
data = loader.load_synthetic_classification()

# Train model
model = RandomForestClassifier()
model.fit(data["X_train"], data["y_train"])

# Create adapter
adapter = SklearnAdapter(model)

# Monitor predictions
result = adapter.predict(data["X_test"], data["y_test"])
print(f"Drift detected: {result['drift']['drift_detected']}")
print(f"Confidence: {result['confidence']['mean_confidence']:.3f}")
```

### PyTorch Integration

```python
from driftwatch.integrations.pytorch_adapter import PyTorchAdapter

# Create a feedforward classifier
model = PyTorchAdapter.create_feedforward_classifier(
    input_dim=10, hidden_dim=64, num_classes=2
)

# Create adapter
adapter = PyTorchAdapter(model)
```

### HuggingFace Integration

```python
from driftwatch.integrations.hf_adapter import HFAdapter

# Uses distilbert-base-uncased-finetuned-sst-2-english
adapter = HFAdapter()

# Monitor text predictions
result = adapter.predict_text([
    "This movie was fantastic!",
    "Terrible experience, would not recommend.",
])
```

---

## Synthetic Drift Generation

DriftWatch includes 8 types of synthetic drift for reproducible experiments:

| Drift Type | Description |
|---|---|
| **Covariate Shift** | Shift feature means |
| **Prior Shift** | Change class balance |
| **Gradual Drift** | Slowly increasing shift |
| **Sudden Drift** | Abrupt point shift |
| **Missingness Drift** | Inject NaN values |
| **Feature Perturbation** | Add noise to subset of features |
| **Gaussian Noise** | Add noise to all features |
| **Feature Corruption** | Set features to constant values |

```python
from driftwatch.data.synthetic_drift import DriftGenerator

generator = DriftGenerator(n_features=5, random_state=42)
reference = generator.generate_reference()
shifted = generator.covariate_shift(n_samples=500, shift_magnitude=2.0)
```

---

## Evaluation

```python
from driftwatch.evaluation.benchmarks import BenchmarkFramework
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector

framework = BenchmarkFramework(detectors=[
    KLDivergenceDetector(),
    PSIDetector(),
])

# Run benchmark
results = framework.run_benchmark(drift_magnitude=2.0)
print(results)

# Sensitivity analysis
sensitivity = framework.run_sensitivity_analysis(
    magnitudes=[0.0, 0.5, 1.0, 2.0, 3.0]
)
```

**Metrics:**
- **Detection latency**: How quickly drift is detected after introduction
- **False positive rate**: Alerts triggered on clean data
- **Detection stability**: Coefficient of variation of scores
- **Sensitivity to drift magnitude**: Score change per unit of drift

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=driftwatch

# Run specific test file
pytest tests/test_detectors.py -v

# Run slow/integration tests
pytest -m "slow"
pytest -m "integration"
```

---

## Research Motivation

DriftWatch was built on the premise that production ML monitoring should go beyond simple drift detection. The **confidence-drift correlation** module addresses a key gap in existing tools:

- **Evidently AI** provides excellent data drift reporting but lacks confidence-drift correlation
- **NannyML** focuses on post-deployment monitoring but doesn't explore lead-lag relationships
- **Why DriftWatch?** We believe confidence degradation can serve as an earlier warning signal than drift scores alone, especially for gradual distribution shifts

See [RESEARCH_NOTE.md](RESEARCH_NOTE.md) for the full academic and theoretical background.

---

## Future Roadmap

- [ ] MC Dropout integration for improved uncertainty estimates
- [ ] Deep kernel learning for MMD-based drift detection
- [ ] Conformal prediction interval monitoring
- [ ] Adaptive threshold calibration
- [ ] REST API for production deployment
- [ ] Kubernetes-native deployment support
- [ ] Real-time alerting (Slack, PagerDuty, email)
- [ ] SQL/NoSQL storage for drift history
- [ ] Multi-model orchestration
- [ ] A/B test monitoring support

---

## Contributing

Contributions are welcome! Please see our contributing guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development setup

```bash
pip install -e ".[dev]"
pytest
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Citation

If you use DriftWatch in your research, please cite:

```bibtex
@software{driftwatch2024,
  author = {DriftWatch Contributors},
  title = {DriftWatch: Real-time Data Drift Detection for ML Systems},
  year = {2024},
  url = {https://github.com/yourusername/driftwatch}
}
```

---

<div align="center">
  <p>Built for production ML reliability · <a href="RESEARCH_NOTE.md">Research Note</a></p>
</div>
