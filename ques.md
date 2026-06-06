# DriftWatch — Q&A

## 1. What is this project about?

DriftWatch is a Python library for **real-time data drift detection** in deployed machine learning systems. It monitors incoming production data, compares it against a training/reference distribution, computes drift metrics over time, and triggers alerts when drift becomes statistically significant.

It goes beyond standard drift detection by introducing a **novel research component**: Confidence-Drift Correlation — determining whether changes in model confidence or uncertainty can serve as earlier warning signals than accuracy degradation.

## 2. What problem does it solve?

**The core problem:** ML models degrade in production. Data distributions shift, user behaviors change, and the world evolves. Most teams only discover this when accuracy metrics start dropping — but by then, the model has been producing poor predictions for potentially thousands or millions of inferences.

**Why this is hard:**
- **Late labels** — ground truth can take days, weeks, or months to arrive
- **Silent failure** — the model still returns predictions, but they become unreliable
- **Gradual degradation** — drift often happens slowly, making it hard to notice

**What DriftWatch does:**
- Detects drift **before** accuracy drops (unsupervised — no labels needed)
- Provides **4 different detection methods** suited for different data types
- Identifies **which features** are shifting
- Monitors **model confidence** as an early warning signal
- Generates **alerts** with severity levels
- Quantifies **early warning effectiveness** through confidence-drift correlation

## 3. Why use this instead of existing tools?

| Feature | DriftWatch | Evidently AI | NannyML |
|---------|:----------:|:------------:|:-------:|
| KL Divergence detector | ✅ | ✅ | ❌ |
| PSI detector | ✅ | ✅ | ✅ |
| MMD detector | ✅ | ❌ | ❌ |
| ADWIN streaming | ✅ | ❌ | ❌ |
| Confidence-drift correlation | ✅ (novel) | ❌ | ❌ |
| Early warning score | ✅ | ❌ | ❌ |
| Lead-lag analysis | ✅ | ❌ | ❌ |
| Per-feature drift ranking | ✅ | ✅ | ✅ |
| Streamlit/FastAPI dashboard | ✅ (FastAPI) | ✅ (Streamlit) | ✅ |
| 8 synthetic drift types | ✅ | ❌ | ❌ |
| Lightweight, pip-installable | ✅ | ✅ | ✅ |
| Research angle (papers cited) | ✅ | ❌ | ❌ |

**DriftWatch's advantage:** It's the only tool that combines standard drift detection with a research-backed confidence-drift correlation module. If you want to know *whether confidence degradation can serve as an early warning*, there's no other off-the-shelf tool that does this.

## 4. Thought process behind this project

### The insight

The project started from a simple observation: **confidence drops before accuracy degrades**.

When you train a neural network, it learns decision boundaries on the training data. When production data starts drifting away from the training distribution, the model enters unfamiliar regions of the feature space. In these regions:

1. The softmax outputs become uncalibrated
2. The model becomes uncertain (high entropy, low confidence)
3. Eventually, predictions become wrong

Step 1 and 2 happen **before** step 3. So if you monitor confidence and entropy, you get an early warning before the model starts failing.

### Why this matters for ML teams

Most production ML systems lack ground truth labels for days or weeks. By the time you compute accuracy, the damage is done. Confidence monitoring gives you an unsupervised, label-free signal that changes continuously — you can set thresholds and get alerts early.

### The design philosophy

- **Simplicity**: Every detector has the same API (`fit`, `score`, `detect`, `summary`). You can swap detectors with a one-line change.
- **Composability**: Detectors are building blocks. Combine them in a StreamMonitor. Add a ConfidenceMonitor. Layer the correlation module on top.
- **Research-backed**: The novel module connects to established work (Gal & Ghahramani's MC Dropout, Guo et al.'s calibration, Hendrycks & Gimpel's OOD detection, Vovk et al.'s conformal prediction).
- **Portfolio-quality**: This was built to demonstrate ML engineering maturity — production-grade packaging, type hints, tests, logging, input validation, dashboard, and documentation.

### The architecture decision

Early on we considered using Streamlit for the dashboard, but it had Python 3.14 compatibility issues. We replaced it with a FastAPI backend + vanilla HTML/CSS/JS frontend with Chart.js. This gives a cleaner separation of concerns, no build step, and a more polished UI experience.

## 5. How to run the project

### Quick start

```bash
# 1. Install the package
pip install -e .

# 2. Run the end-to-end demo
python demo.py

# 3. Or run a quick 15-batch version
python demo.py --quick
```

### Run benchmarks

```bash
python benchmarks.py
```

This runs 6 benchmarks comparing all detectors across:
- Detection accuracy (FPR, latency, stability)
- Sensitivity to drift magnitude
- Performance across drift types
- ADWIN streaming effectiveness
- Confidence early-warning score
- PSI per-feature analysis

### Launch the dashboard

```bash
# Via the demo
python demo.py --dashboard

# Directly
python -m driftwatch.dashboard.server

# Then open http://localhost:8501
```

The dashboard has 6 pages:
1. **Overview** — system status, alerts, recent drift scores
2. **Drift Monitoring** — PSI, MMD, KL, ADWIN trends with toggles
3. **Feature-Level Analysis** — per-feature drift heatmap and ranking
4. **Confidence Monitoring** — confidence, entropy, margin history
5. **Confidence-Drift Correlation** — lead-lag plots and early warning score
6. **Alerts** — filterable alert log with severity breakdown

### Run tests

```bash
# All tests
pytest

# With coverage
pytest --cov=driftwatch

# Specific test file
pytest tests/test_detectors.py -v
```

### Export results

```bash
# Export to JSON
python demo.py --export results.json

# Export drift scores + confidence to CSV
python demo.py --export-csv scores.csv
```

### Use in your own code

```python
import numpy as np
from driftwatch.monitors.stream_monitor import StreamMonitor

# Create a monitor
monitor = StreamMonitor()

# Fit on your reference (training) data
reference = np.random.normal(0, 1, (500, 3))
monitor.fit(reference)

# Process production batches
for batch_idx in range(10):
    batch = np.random.normal(batch_idx * 0.2, 1, (100, 3))
    result = monitor.process_batch(batch)
    print(f"Batch {batch_idx}: {result['status']}")
```

### Notebook

```bash
jupyter notebook notebooks/driftwatch_case_study.ipynb
```

The notebook provides a complete walkthrough covering data generation, drift detection across all 4 detectors, confidence monitoring, confidence-drift correlation, and visualization.

## 6. Project structure

```
driftwatch/
+-- detectors/        # KL, PSI, MMD, ADWIN — all with unified API
+-- monitors/         # StreamMonitor, ConfidenceMonitor
+-- alerts/           # Alert schemas, threshold/rolling rules
+-- correlation/      # Novel confidence-drift correlation module
+-- data/             # 8 synthetic drift types, data loaders
+-- integrations/     # sklearn, PyTorch, HuggingFace adapters
+-- evaluation/       # Metrics, benchmarks, comparison framework
+-- dashboard/        # FastAPI server + static frontend (Chart.js)
+-- utils/            # Validation, logging, stats helpers
```

## 7. Key metrics

After running the demo (25 batches, drift starts at batch 8):

| Metric | Typical value |
|--------|:-------------:|
| Early warning score | 50-65/100 |
| Confidence drop | 17-26% |
| Drift score increase | 10-60x (varies by detector) |
| Detection latency | 0-2 batches |
| PSI aggregate (heavily drifted) | 9-17 |
| MMD (heavily drifted) | 0.33-0.45 |
| Alerts triggered | 54-90 per run |
