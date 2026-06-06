# Getting Started

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

- Python >= 3.9
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
print(f"KL Divergence: {result['score']:.4f} - Drift: {result['drift_detected']}")

# PSI Detector
psi_detector = PSIDetector(threshold=0.1)
psi_detector.fit(reference)
result = psi_detector.detect(production)
print(f"PSI: {result['score']:.4f} - Drift: {result['drift_detected']}")

# MMD Detector
mmd_detector = MMDDetector(threshold=0.05)
mmd_detector.fit(reference)
result = mmd_detector.detect(production)
print(f"MMD: {result['score']:.4f} - Drift: {result['drift_detected']}")
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

### Run the demo

```bash
# Full 25-batch demo
python demo.py

# Quick 15-batch demo
python demo.py --quick

# Export results to JSON
python demo.py --export results.json

# Export drift scores to CSV
python demo.py --export-csv scores.csv

# Launch Streamlit dashboard after demo
python demo.py --dashboard
```

---

## Project Structure

```
driftwatch/
+-- __init__.py              # Package init with version
+-- detectors/               # Drift detection algorithms
|   +-- base.py              # Abstract base class
|   +-- kl.py                # KL Divergence
|   +-- psi.py               # Population Stability Index
|   +-- mmd.py               # Maximum Mean Discrepancy
|   +-- adwin.py             # ADWIN-style adaptive windowing
+-- monitors/                # High-level monitors
|   +-- stream_monitor.py    # Batch ingestion and coordination
|   +-- confidence_monitor.py # Confidence, entropy, margin tracking
+-- alerts/                  # Alerting system
|   +-- schemas.py           # Alert dataclass, Severity enum
|   +-- rules.py             # ThresholdRule, RollingWindowRule, AlertEngine
+-- correlation/             # Novel research module
|   +-- confidence_drift.py  # Confidence-drift correlation analysis
+-- data/                    # Data utilities
|   +-- synthetic_drift.py   # 8 drift types for reproducible experiments
|   +-- loaders.py           # Dataset loaders for demos
+-- integrations/            # Model integration adapters
|   +-- sklearn_adapter.py   # Scikit-learn wrapper
|   +-- pytorch_adapter.py   # PyTorch wrapper
|   +-- hf_adapter.py        # HuggingFace DistilBERT wrapper
+-- evaluation/              # Evaluation framework
|   +-- metrics.py           # Latency, FPR, stability, sensitivity
|   +-- benchmarks.py        # Benchmark framework + external tool compat
+-- dashboard/               # Streamlit visualization
|   +-- app.py               # Multi-page dashboard
|   +-- visuals.py           # Plotly visualization helpers
+-- utils/                   # Shared utilities
    +-- validation.py        # Input validation
    +-- logging.py           # Logging configuration
    +-- stats.py             # Statistical helpers
```
