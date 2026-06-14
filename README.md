# Production Drift Detection
### Real-Time Data Drift Detection for Production ML Systems

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Dashboard-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" />
</p>

> A lightweight, pip-installable Python library for monitoring data drift and model confidence in production ML systems. Detects when real-world data begins diverging from the training distribution - before model performance visibly degrades.

---

## Table of Contents

- [The Problem](#the-problem)
- [What This Does](#what-this-does)
- [Benchmark Results](#benchmark-results)
- [Research: Confidence-Drift Correlation](#research-confidence-drift-correlation)
- [Drift Detection Methods](#drift-detection-methods)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Confidence Monitoring](#confidence-monitoring)
- [Model Integrations](#model-integrations)
- [Synthetic Drift Generation](#synthetic-drift-generation)
- [Dashboard](#dashboard)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Related Work](#related-work)
- [Citation](#citation)

---

## The Problem

Ground truth labels arrive late in almost every production ML system:

- Credit risk: defaults take months to confirm
- Medical diagnosis: pathology results take days to weeks
- Fraud detection: chargebacks take weeks to materialize
- Recommendations: satisfaction is measured indirectly

By the time accuracy degradation is confirmed, the model may have been producing poor predictions for thousands of inferences. **Unsupervised monitoring - detecting degradation without labels - is essential for production ML reliability.**

---

## What This Does

Production Drift Detection provides:

1. **Four drift detectors** (KL Divergence, PSI, MMD, ADWIN) with a unified `fit/score/detect/summary` API
2. **Stream monitoring** with batch ingestion and rolling statistics
3. **Confidence monitoring** tracking entropy, margin, and trend signals
4. **Confidence-Drift Correlation module** - empirically testing whether confidence degradation precedes drift detection
5. **Alerting system** with four severity levels: Healthy, Watch, Warning, Critical
6. **Interactive 6-page dashboard** (FastAPI + Chart.js)
7. **Synthetic drift generator** with 8 drift types for reproducible experiments
8. **Model integrations** for Scikit-learn, PyTorch, and HuggingFace

---

## Benchmark Results

Rigorous benchmark across **20 random seeds**, 5 features, 2000 reference samples, drift magnitude 2.0 (10-batch ramp + 40 full-strength batches). All metrics reported as mean ± 2×SEM (~95% CI).

### Core Metrics

| Detector | FPR (clean) | Detection Rate | ROC AUC | Cohen's d (strong) | Composite Rank |
|---|---|---|---|---|---|
| **MMD** | **0.0%** | 99.9% ± 0.2% | **1.0000** | 6.38 ± 0.43 | **#1 (0.9225)** |
| PSI | 39.9% ± 3.0% | **100.0%** | **1.0000** | 4.62 ± 0.26 | #2 (0.8537) |
| KL Divergence | **0.0%** | 95.2% ± 1.6% | 0.9995 ± 0.0008 | 1.18 ± 0.11 | #3 (0.7831) |
| ADWIN | 46.5% ± 6.2% | 97.7% ± 1.0% | 0.9751 ± 0.0079 | 2.37 ± 0.13 | #4 (0.7187) |

### Detection Latency

All four detectors achieve **median latency of 0 batches** - drift is flagged in the same batch it begins. KL Divergence averages 0.20 batches (max 1 batch across all seeds), making all methods effectively immediate.

| Detector | Median Latency | Mean Latency | % Detected |
|---|---|---|---|
| KL Divergence | 0 batches | 0.20 batches | 100% |
| PSI | 0 batches | 0.00 batches | 100% |
| MMD | 0 batches | 0.00 batches | 100% |
| ADWIN | 0 batches | 0.00 batches | 100% |

### Threshold Calibration (Youden's F1-Max)

Default thresholds are well-calibrated for KL and MMD. PSI and ADWIN benefit from recalibration:

| Detector | Default Threshold | Optimal Threshold | F1 (default) | F1 (optimal) | Gain |
|---|---|---|---|---|---|
| KL Divergence | 0.1000 | 0.1000 | 0.9708 | 0.9708 | +0.0000 |
| PSI | 0.1000 | 0.1898 | 0.7567 | 0.9499 | **+0.1932** |
| MMD | 0.0500 | 0.0108 | 0.9994 | **1.0000** | +0.0006 |
| ADWIN | 0.1000 | 0.3778 | 0.7700 | 0.9573 | **+0.1872** |

**Recommendation:** Use MMD as the primary detector (zero FPR, perfect AUC, immediate detection). Pair with KL Divergence as a secondary signal. Recalibrate PSI and ADWIN thresholds before production deployment.

---

## Research: Confidence-Drift Correlation

### Hypothesis

> **H1:** Under gradual covariate shift, changes in model confidence precede changes in drift scores by k time steps.
>
> **H2:** The lead-lag relationship is asymmetric - confidence is more likely to lead drift than vice versa.

### Empirical Validation

Tested across **30 trials** (3 experiment types x 10 seeds), two model classes (LogisticRegression, PyTorch NN), two drift types (covariate shift, perturbation).

| Experiment | H1 Support (KL) | H1 Support (PSI) | H1 Support (MMD) | Avg EWS | Conf Drop | Acc Drop |
|---|---|---|---|---|---|---|
| sklearn - covariate shift | 4/10 (40%) | 3/10 (30%) | 2/10 (20%) | 42/100 | -2.69% | -1.45% |
| PyTorch NN - covariate shift | 3/10 (30%) | 3/10 (30%) | 3/10 (30%) | 42/100 | -1.42% | +6.01% |
| sklearn - perturbation | 2/10 (20%) | 5/10 (50%) | 1/10 (10%) | 44/100 | -2.35% | -1.45% |

### Conclusion

**H1 is not supported.** Confidence precedes drift in only 20-40% of trials depending on detector and model class - not reliably enough to serve as a universal early warning signal. Across all 30 trials, roughly 27-30% of detector-seed pairs show confidence leading drift.

This is an honest empirical result. The confidence-drift lead-lag relationship is **detector-dependent and non-deterministic** under gradual covariate shift. The Confidence-Drift Correlation module remains useful as a diagnostic and monitoring signal, but should not be relied upon as a consistent early tripwire without further investigation into conditions where leading behaviour emerges.

Cross-correlation analysis does show meaningful **co-movement** between confidence and drift signals (max correlations 0.38-0.77 across detectors), confirming that confidence tracks drift even when it does not reliably precede it.

```python
from production-drift-detection.correlation.confidence_drift import ConfidenceDriftCorrelation

correlator = ConfidenceDriftCorrelation(max_lag=10)
result = correlator.analyze(confidence_history, drift_history)

print(f"Lead-lag: {result['lead_lag_steps']} steps")
print(f"Early warning score: {result['early_warning_score']}/100")
```

---

## Drift Detection Methods

### KL Divergence
Measures divergence between reference and current probability distributions. Best for categorical features and probability outputs. Laplace smoothing for numerical stability. **Zero FPR in benchmarks, ROC AUC 0.9995.** Default threshold (0.1) is optimally calibrated - no recalibration needed.

### Population Stability Index (PSI)
Measures feature distribution stability over time using binned proportions. Standard convention: PSI < 0.1 (stable), 0.1-0.25 (moderate shift), > 0.25 (significant shift). **Perfect detection rate (100%) but high FPR (39.9%)** - recalibrate threshold to 0.1898 for production use.

### Maximum Mean Discrepancy (MMD)
Kernel-based two-sample test using RBF kernels. Supports multivariate distributions. Median heuristic for automatic bandwidth selection. **Top-ranked detector: zero FPR, perfect AUC (1.0000), Cohen's d of 6.38 under strong drift.** Optimal threshold is 0.0108, not the default 0.05.

### ADWIN-Style Adaptive Windowing
Online drift detection for streaming data. Adaptive window resizing based on Hoeffding-bound change detection. No need to store full historical data. **High FPR (46.5%) and weaker perturbation AUC (0.65)** - requires threshold recalibration to 0.3778. Best suited for scenarios requiring online, memory-efficient detection.

---

## Architecture

```
Production Data Stream
         |
         v
+---------------------+
|   Stream Monitor    |   Batch ingestion + rolling statistics
|                     |   Coordinates all detectors
+----------+----------+
           |
    +------+------+
    v             v
+---------+  +------------------+
| Drift   |  |   Confidence     |
|Detectors|  |   Monitor        |
|KL/PSI/  |  | Entropy, Margin, |
|MMD/ADWIN|  | Trend Analysis   |
+----+----+  +--------+---------+
     |                |
     +-------+--------+
             |
             v
+---------------------+
| Confidence-Drift    |   Lead-lag correlation analysis
| Correlation Module  |   Early warning scoring
+----------+----------+
           |
           v
+---------------------+
|   Alert Engine      |   Threshold + rolling window rules
|                     |   Healthy / Watch / Warning / Critical
+----------+----------+
           |
           v
    Dashboard + Alerts
```

---

## Repository Structure

```
production-drift-detection/
|
+-- src/                         # Core library source
|   +-- detectors/               # KL, PSI, MMD, ADWIN
|   +-- monitors/                # StreamMonitor, ConfidenceMonitor
|   +-- alerts/                  # Alert schemas, rules, engine
|   +-- correlation/             # Confidence-drift correlation module
|   +-- data/                    # Synthetic drift generator, loaders
|   +-- integrations/            # Sklearn, PyTorch, HuggingFace adapters
|   +-- evaluation/              # Metrics, benchmarks
|   +-- dashboard/               # FastAPI backend + Chart.js frontend
|   +-- utils/                   # Validation, logging, statistics
|
+-- notebooks/                   # Experiment notebooks
+-- tests/                       # Test suite
+-- benchmarks.py                # Standard benchmark runner
+-- benchmark_rigorous.py        # 20-seed rigorous benchmark suite
+-- empirical_validation.py      # Confidence-drift correlation validation
+-- demo.py                      # Demo + dashboard launcher
+-- pyproject.toml
+-- LICENSE
```

---

## Installation

```bash
# From source
git clone https://github.com/royxlead/production-drift-detection.git
cd production-drift-detection
pip install -e .

# With optional dependencies
pip install -e ".[pytorch]"        # PyTorch support
pip install -e ".[transformers]"   # HuggingFace support
pip install -e ".[dev]"            # Testing + notebooks
pip install -e ".[all]"            # Everything
```

**Requirements:** Python 3.9+ · NumPy 2.4+ · Pandas 3.0+

---

## Quick Start

```python
import numpy as np
from production-drift-detection.detectors.mmd import MMDDetector      # Recommended
from production-drift-detection.detectors.kl import KLDivergenceDetector
from production-drift-detection.detectors.psi import PSIDetector

# Reference distribution (training data)
reference = np.random.normal(0, 1, (2000, 5))
# Production data (shifted)
production = np.random.normal(2, 1, (500, 5))

# Unified API across all detectors
detector = MMDDetector(threshold=0.0108)   # Use calibrated threshold
detector.fit(reference)
result = detector.detect(production)

print(f"Drift detected: {result['drift_detected']}")
print(f"Score: {result['score']:.4f}")
```

**Stream monitoring:**

```python
from production-drift-detection.monitors.stream_monitor import StreamMonitor

monitor = StreamMonitor()
monitor.fit(reference)

for batch_idx in range(10):
    batch = np.random.normal(batch_idx * 0.2, 1, (100, 5))
    result = monitor.process_batch(batch)
    print(f"Batch {batch_idx}: Status={result['status']}")
```

---

## Confidence Monitoring

```python
from production-drift-detection.monitors.confidence_monitor import ConfidenceMonitor

monitor = ConfidenceMonitor()

for batch_predictions in prediction_stream:
    status = monitor.update(batch_predictions)
    print(f"Mean confidence: {status['mean_confidence']:.3f}")
    print(f"Entropy trend: {status['entropy_trend']}")
    print(f"Degradation detected: {status['degradation_detected']}")
```

**Tracked metrics:** Mean confidence, predictive entropy, margin (top-2 gap), trend direction, ECE approximation, over/underconfidence ratios.

---

## Model Integrations

**Scikit-learn:**

```python
from production-drift-detection.integrations.sklearn_adapter import SklearnAdapter

adapter = SklearnAdapter(sklearn_model)
result = adapter.predict(X_test, y_test)
print(f"Drift: {result['drift']['drift_detected']}")
print(f"Confidence: {result['confidence']['mean_confidence']:.3f}")
```

**PyTorch:**

```python
from production-drift-detection.integrations.pytorch_adapter import PyTorchAdapter

adapter = PyTorchAdapter(torch_model)
result = adapter.predict(X_tensor)
```

**HuggingFace:**

```python
from production-drift-detection.integrations.hf_adapter import HFAdapter

adapter = HFAdapter()   # DistilBERT SST-2 by default
result = adapter.predict_text(["Great product!", "Terrible experience."])
```

---

## Synthetic Drift Generation

8 drift types for reproducible research:

| Drift Type | Description |
|---|---|
| Covariate Shift | Shift feature means |
| Prior Shift | Change class balance |
| Gradual Drift | Slowly increasing shift over time |
| Sudden Drift | Abrupt point shift |
| Missingness Drift | Inject NaN values |
| Feature Perturbation | Add noise to a subset of features |
| Gaussian Noise | Add noise to all features |
| Feature Corruption | Set features to constant values |

```python
from production-drift-detection.data.synthetic_drift import DriftGenerator

generator = DriftGenerator(n_features=5, random_state=42)
reference = generator.generate_reference(n_samples=2000)
shifted = generator.gradual_drift(n_samples=500, drift_magnitude=2.0)
```

---

## Dashboard

```bash
python -m production-drift-detection.dashboard.server
# or
python demo.py --dashboard
```

**6 pages:**
1. Overview - current status, active alerts, recent scores
2. Drift Monitoring - PSI, MMD, KL, ADWIN score trends
3. Feature Analysis - per-feature drift heatmap
4. Confidence Monitoring - confidence, entropy, margin history
5. Confidence-Drift Correlation - lead-lag plots, cross-correlation
6. Alerts - filterable log with severity indicators

---

## Evaluation

Run the rigorous benchmark (20 seeds, ~45 seconds):

```bash
python benchmark_rigorous.py
```

Run the confidence-drift empirical validation:

```bash
python empirical_validation.py
```

Run standard benchmarks programmatically:

```python
from production-drift-detection.evaluation.benchmarks import BenchmarkFramework
from production-drift-detection.detectors.kl import KLDivergenceDetector
from production-drift-detection.detectors.mmd import MMDDetector

framework = BenchmarkFramework(detectors=[
    KLDivergenceDetector(),
    MMDDetector(),
])

results = framework.run_benchmark(drift_magnitude=2.0)
sensitivity = framework.run_sensitivity_analysis(
    magnitudes=[0.0, 0.5, 1.0, 2.0, 3.0]
)
```

**Testing:**

```bash
pytest                            # All tests
pytest --cov=production-drift-detection           # With coverage
pytest tests/test_detectors.py   # Specific module
pytest -m "integration"          # Integration tests only
```

---

## Research Background

Production Drift Detection connects to established literature on distribution shift and uncertainty estimation:

- **Confidence-accuracy gap under shift** - Guo et al. (2017) showed modern networks are systematically overconfident. Under distribution shift, ECE increases before accuracy degrades.
- **Entropy as uncertainty signal** - High entropy predictions signal OOD inputs. Production Drift Detection tracks entropy trends as a population-level monitoring signal.
- **Self-diagnosing models** - Leibig et al. (2017) demonstrated neural networks can estimate their own uncertainty. Production Drift Detection operationalizes this at the monitoring layer.
- **Bayesian uncertainty** - MC Dropout (Gal and Ghahramani, 2016) approximates epistemic uncertainty. Production Drift Detection's ConfidenceMonitor is designed to be compatible with MC Dropout outputs.

**Why confidence tracks drift even without leading it:**
Confidence is continuous - it responds to small perturbations. Accuracy is discontinuous - a prediction is right or wrong. Both signals are informative; their relationship is model-dependent and drift-type-dependent rather than universally ordered.

---

## Roadmap

- [ ] MC Dropout integration for improved epistemic uncertainty estimates
- [ ] Deep kernel learning for MMD-based drift detection
- [ ] Conformal prediction interval monitoring
- [ ] Adaptive threshold calibration (auto-recalibration from validation data)
- [ ] REST API for production deployment
- [ ] Kubernetes-native deployment support
- [ ] Real-time alerting (Slack, PagerDuty, email)
- [ ] SQL/NoSQL storage for drift history
- [ ] Multi-model orchestration
- [ ] A/B test monitoring support

---

## Related Work

- [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python) - Unsupervised confidence estimation for neural networks
- [CURA](https://github.com/royxlead/cura-python) - RAG reliability and hallucination mitigation
- [AutoLLM Forge](https://github.com/royxlead/autollmforge-python) - Efficient LLM fine-tuning

---

## Citation

```bibtex
@software{roy2026production-drift-detection,
  author = {Roy, Sourav},
  title  = {Production Drift Detection: Real-Time Data Drift Detection for Production ML Systems},
  year   = {2026},
  url    = {https://github.com/royxlead/production-drift-detection}
}
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/royxlead">Sourav Roy</a> · Founding AI/ML Engineer · Yuga AI</sub>
</p>