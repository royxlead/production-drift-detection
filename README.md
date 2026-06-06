<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=120&section=header&text=DriftWatch&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Real-Time%20Data%20Drift%20Detection%20for%20Production%20ML&descAlignY=60&descSize=15&descColor=a5b4fc" width="100%"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Sklearn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/Dashboard-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

## Overview

DriftWatch is a lightweight, pip-installable Python library for monitoring data drift and model confidence in production ML systems. It detects when real-world data begins diverging from the training distribution - before model performance visibly degrades.

**The core research question driving this project:**

> *Can changes in model confidence serve as earlier warning signals than drift scores or accuracy degradation - and can we measure the lead-lag relationship systematically?*

Most production ML monitoring tools detect drift after it has already affected model outputs. DriftWatch introduces a **Confidence-Drift Correlation** module that tests whether confidence degradation consistently precedes drift detection - providing actionable lead time before accuracy drops.

---

## The Problem: Late Labels in Production

Ground truth labels arrive late in almost every production ML system:

- Credit risk models: defaults take months to confirm
- Medical diagnosis: pathology results take days to weeks
- Fraud detection: chargebacks take weeks to materialize
- Recommendation systems: satisfaction is measured indirectly

By the time accuracy degradation is confirmed, the model may have been producing poor predictions for thousands of inferences. **Unsupervised monitoring - detecting degradation without labels - is essential for production ML reliability.**

---

## Key Features

| Feature | Description |
|---|---|
| **4 Drift Detectors** | KL Divergence, PSI, MMD, ADWIN-style adaptive windowing |
| **Unified API** | Consistent `fit/score/detect/summary` interface across all detectors |
| **Stream Monitoring** | Batch ingestion with rolling statistics |
| **Confidence Monitoring** | Entropy, margin, and trend analysis |
| **Confidence-Drift Correlation** | Novel lead-lag analysis - does confidence degrade before drift? |
| **Alerting System** | Severity levels: Healthy, Watch, Warning, Critical |
| **Interactive Dashboard** | 6-page FastAPI + Chart.js visualization suite |
| **Synthetic Drift Generator** | 8 drift types for reproducible experiments |
| **Model Integrations** | Scikit-learn, PyTorch, HuggingFace |
| **Evaluation Framework** | Detection latency, FPR, stability, sensitivity metrics |

---

## Installation

```bash
# From source
git clone https://github.com/royxlead/driftwatch.git
cd driftwatch
pip install -e .

# With optional dependencies
pip install -e ".[pytorch]"       # PyTorch support
pip install -e ".[transformers]"  # HuggingFace support
pip install -e ".[dev]"           # Testing + notebooks
pip install -e ".[all]"           # Everything
```

---

## Quick Start

```python
import numpy as np
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.detectors.mmd import MMDDetector

# Reference distribution (training data)
reference = np.random.normal(0, 1, (500, 3))
# Production data (shifted)
production = np.random.normal(2, 1, (500, 3))

# Any detector - same API
detector = MMDDetector(threshold=0.05)
detector.fit(reference)
result = detector.detect(production)

print(f"Drift detected: {result['drift_detected']}")
print(f"Score: {result['score']:.4f}")
```

**Stream monitoring:**

```python
from driftwatch.monitors.stream_monitor import StreamMonitor

monitor = StreamMonitor()
monitor.fit(reference)

for batch_idx in range(10):
    batch = np.random.normal(batch_idx * 0.2, 1, (100, 3))
    result = monitor.process_batch(batch)
    print(f"Batch {batch_idx}: Status={result['status']}")
```

---

## Architecture

```
Production Data Stream
         │
         ▼
┌─────────────────────┐
│   Stream Monitor    │   Batch ingestion + rolling statistics
│                     │   Coordinates all detectors
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────────────┐
│ Drift  │  │   Confidence     │
│Detectors│  │   Monitor        │
│KL/PSI/ │  │ Entropy, Margin, │
│MMD/ADWIN│  │ Trend Analysis   │
└────┬───┘  └────────┬─────────┘
     │               │
     └───────┬───────┘
             │
             ▼
┌─────────────────────┐
│ Confidence-Drift    │   Novel lead-lag correlation
│ Correlation Module  │   Early warning scoring
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Alert Engine      │   Threshold + rolling window rules
│                     │   Severity: Healthy/Watch/Warning/Critical
└──────────┬──────────┘
           │
           ▼
    Dashboard + Alerts
```

---

## Drift Detection Methods

**KL Divergence** - Measures divergence between reference and current probability distributions. Best for categorical features and probability outputs. Laplace smoothing for numerical stability.

**Population Stability Index (PSI)** - Measures feature distribution stability over time using binned proportions. Standard convention: PSI < 0.1 (stable), 0.1 to 0.25 (moderate), > 0.25 (significant shift).

**Maximum Mean Discrepancy (MMD)** - Kernel-based two-sample test using RBF kernels. Supports multivariate distributions. Median heuristic for automatic bandwidth selection.

**ADWIN-Style Adaptive Windowing** - Online drift detection for streaming data. Adaptive window resizing based on Hoeffding-bound change detection. No need to store full historical data.

---

## Novel Research: Confidence-Drift Correlation

The differentiating module. Addresses a key research hypothesis:

> **H1:** Under gradual covariate shift, changes in model confidence precede changes in drift scores by k time steps.
>
> **H2:** The lead-lag relationship is asymmetric - confidence is more likely to lead drift than vice versa.

**If H1 and H2 hold:**
- Confidence monitoring provides lead time before drift scores reach alert thresholds
- Combined confidence-drift alerts are more robust than either signal alone
- Tighter alert thresholds on confidence can serve as earlier tripwires

**Implementation:**

```python
from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation

correlator = ConfidenceDriftCorrelation(max_lag=10)
result = correlator.analyze(confidence_history, drift_history)

print(f"Lead-lag: {result['lead_lag_steps']} steps")
print(f"Early warning score: {result['early_warning_score']}/100")
```

**Early Warning Score** combines:
- Confidence degradation rate
- Drift acceleration
- Leading indicator status

---

## Confidence Monitoring

```python
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor

monitor = ConfidenceMonitor()

# Track predictions over time
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
from driftwatch.integrations.sklearn_adapter import SklearnAdapter

adapter = SklearnAdapter(sklearn_model)
result = adapter.predict(X_test, y_test)
print(f"Drift: {result['drift']['drift_detected']}")
print(f"Confidence: {result['confidence']['mean_confidence']:.3f}")
```

**PyTorch:**

```python
from driftwatch.integrations.pytorch_adapter import PyTorchAdapter

adapter = PyTorchAdapter(torch_model)
result = adapter.predict(X_tensor)
```

**HuggingFace:**

```python
from driftwatch.integrations.hf_adapter import HFAdapter

adapter = HFAdapter()  # DistilBERT SST-2 by default
result = adapter.predict_text(["Great product!", "Terrible experience."])
```

---

## Synthetic Drift Generation

8 drift types for reproducible research:

| Drift Type | Description |
|---|---|
| Covariate Shift | Shift feature means |
| Prior Shift | Change class balance |
| Gradual Drift | Slowly increasing shift |
| Sudden Drift | Abrupt point shift |
| Missingness Drift | Inject NaN values |
| Feature Perturbation | Add noise to subset of features |
| Gaussian Noise | Add noise to all features |
| Feature Corruption | Set features to constant values |

```python
from driftwatch.data.synthetic_drift import DriftGenerator

generator = DriftGenerator(n_features=5, random_state=42)
reference = generator.generate_reference()
shifted = generator.gradual_drift(n_samples=500, drift_magnitude=2.0)
```

---

## Dashboard

```bash
python -m driftwatch.dashboard.server
# or
python demo.py --dashboard
```

**6 pages:**
1. Overview - current status, active alerts, recent scores
2. Drift Monitoring - PSI, MMD, KL, ADWIN trends
3. Feature Analysis - per-feature drift heatmap
4. Confidence Monitoring - confidence, entropy, margin history
5. Confidence-Drift Correlation - lead-lag plots, cross-correlation
6. Alerts - filterable log with severity indicators

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

results = framework.run_benchmark(drift_magnitude=2.0)
sensitivity = framework.run_sensitivity_analysis(
    magnitudes=[0.0, 0.5, 1.0, 2.0, 3.0]
)
```

**Metrics:** Detection latency, false positive rate, detection stability, sensitivity to drift magnitude.

---

## Testing

```bash
pytest                          # All tests
pytest --cov=driftwatch         # With coverage
pytest tests/test_detectors.py  # Specific module
pytest -m "integration"         # Integration tests
```

---

## Repository Structure

```
driftwatch/
├── detectors/          # KL, PSI, MMD, ADWIN
├── monitors/           # StreamMonitor, ConfidenceMonitor
├── alerts/             # Alert schemas, rules, engine
├── correlation/        # Confidence-drift correlation (novel)
├── data/               # Synthetic drift generator, loaders
├── integrations/       # Sklearn, PyTorch, HuggingFace adapters
├── evaluation/         # Metrics, benchmarks
├── dashboard/          # FastAPI backend + Chart.js frontend
└── utils/              # Validation, logging, statistics
```

---

## Research Background

DriftWatch connects to an established literature on uncertainty estimation and distribution shift:

- **Confidence-accuracy gap under shift** - Guo et al. (2017) showed modern networks are systematically overconfident. Under distribution shift, ECE increases before accuracy degrades.
- **Entropy as uncertainty signal** - High entropy predictions signal OOD inputs. DriftWatch tracks entropy trends as a population-level early warning.
- **Self-diagnosing models** - Leibig et al. (2017) demonstrated neural networks can estimate their own uncertainty. DriftWatch operationalizes this at the monitoring layer.
- **Bayesian uncertainty** - MC Dropout (Gal & Ghahramani, 2016) approximates epistemic uncertainty. DriftWatch's ConfidenceMonitor is designed to be compatible with MC Dropout outputs.

**Why confidence can degrade before accuracy:**
1. Confidence is continuous - responds to small perturbations
2. Accuracy is discontinuous - a prediction is right or wrong
3. Models become uncertain about boundary cases before crossing into consistent error

---

## Roadmap

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

## Related Work

- [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python) - Published research on unsupervised confidence estimation
- [CURA](https://github.com/royxlead/cura-python) - RAG reliability and hallucination mitigation
- [AutoLLM Forge](https://github.com/royxlead/autollmforge-python) - Efficient LLM fine-tuning

---

## Citation

```bibtex
@software{driftwatch2025,
  author = {Roy, Sourav},
  title = {DriftWatch: Real-time Data Drift Detection for Production ML Systems},
  year = {2025},
  url = {https://github.com/royxlead/driftwatch}
}
```

---

<div align="center">

**[Portfolio](https://royxlead.netlify.app) · [LinkedIn](https://linkedin.com/in/royxlead) · [ORCID](https://orcid.org/0009-0009-6582-2295)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=80&section=footer" width="100%"/>

</div>