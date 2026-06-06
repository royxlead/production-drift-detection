# Case Study Notebook

DriftWatch includes a comprehensive Jupyter notebook that serves as a complete walkthrough for researchers, ML engineers, and graduate admissions reviewers.

## Running the Notebook

```bash
# Navigate to the project root
cd driftwatch

# Launch Jupyter
jupyter notebook notebooks/driftwatch_case_study.ipynb

# Or with JupyterLab
jupyter lab notebooks/driftwatch_case_study.ipynb
```

## Notebook Contents

The notebook demonstrates a full end-to-end drift detection workflow:

### 1. Setup and Imports
- Installing and importing DriftWatch components
- Setting random seeds for reproducibility

### 2. Data Generation
- Creating a reference (training) distribution
- Generating production batches with increasing drift
- Visualizing the drift injection process

### 3. Drift Detection
- **PSI Monitoring**: Track population stability over time
- **KL Divergence**: Detect categorical distribution shifts
- **MMD**: Kernel-based multivariate drift detection
- **ADWIN**: Adaptive streaming drift detection

### 4. Confidence Monitoring
- Tracking prediction confidence over time
- Monitoring entropy as uncertainty signal
- Detecting confidence degradation

### 5. Confidence-Drift Correlation
- Computing cross-correlation between confidence and drift
- Estimating lead-lag relationships
- Generating early warning scores
- Visualizing the correlation structure

### 6. Alerting and Reporting
- Threshold-based alert configuration
- Alert history and severity tracking
- JSON/CSV result export

### 7. Visualization
- Drift score trends across all detectors
- Feature-level drift heatmaps
- Confidence history plots
- Lead-lag correlation plots
- Alert timeline

## Expected Outcomes

After running the notebook, you should observe:

1. **Drift scores increase** as drift magnitude grows
2. **Confidence degrades** before drift becomes severe (the key research finding)
3. **Alerts trigger** at appropriate severity levels
4. **Cross-correlation** shows confidence leading drift by 1-3 batches
5. **Early warning score** quantifies the lead-lag relationship

## Customization

The notebook is designed to be easily customizable:

```python
# Change the number of features
generator = DriftGenerator(n_features=10, random_state=42)

# Change drift intensity
batches = generator.generate_demo_pipeline(
    n_batches=30,
    batch_size=200,
    drift_start_batch=10,
    drift_increment=0.2,
)

# Use different detectors
monitor = StreamMonitor(detectors={
    "psi": PSIDetector(threshold=0.05),
    "mmd": MMDDetector(threshold=0.02),
})
```
