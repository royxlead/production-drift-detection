"""Tests for dashboard data preparation."""

import numpy as np
import pytest

from production_drift_detection.dashboard.visuals import Visualizer
from production_drift_detection.data.synthetic_drift import DriftGenerator
from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor


class TestVisualizer:
    """Tests for the dashboard Visualizer."""

    def test_plot_drift_scores(self):
        viz = Visualizer()
        history = {
            "kl": [0.05, 0.08, 0.12, 0.20],
            "psi": [0.03, 0.06, 0.10, 0.18],
        }
        fig = viz.plot_drift_scores(history)
        assert fig is not None
        assert len(fig.data) == 2

    def test_plot_feature_drift_heatmap(self):
        viz = Visualizer()
        feature_scores = [
            {"a": 0.1, "b": 0.05, "c": 0.2},
            {"a": 0.15, "b": 0.08, "c": 0.25},
        ]
        fig = viz.plot_feature_drift_heatmap(feature_scores, ["a", "b", "c"])
        assert fig is not None

    def test_plot_confidence_history_empty(self):
        viz = Visualizer()
        monitor = ConfidenceMonitor()
        fig = viz.plot_confidence_history(monitor)
        assert fig is not None

    def test_plot_confidence_history_with_data(self):
        viz = Visualizer()
        monitor = ConfidenceMonitor()
        for _ in range(5):
            probs = np.random.dirichlet([1, 9], size=50)
            monitor.update(probs)
        fig = viz.plot_confidence_history(monitor)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_detection_latency(self):
        viz = Visualizer()
        scores = np.array([0.05, 0.04, 0.06, 0.15, 0.25, 0.30])
        fig = viz.plot_detection_latency(scores, threshold=0.1, drift_start_idx=3)
        assert fig is not None

    def test_plot_drift_summary(self):
        viz = Visualizer()
        gen = DriftGenerator(n_features=3, random_state=42)

        from production_drift_detection.detectors.kl import KLDivergenceDetector
        from production_drift_detection.detectors.psi import PSIDetector

        kl_det = KLDivergenceDetector()
        psi_det = PSIDetector()

        reference = gen.generate_reference()
        kl_det.fit(reference)
        psi_det.fit(reference)

        summary = {
            "detectors": {
                "kl": kl_det.summary(),
                "psi": psi_det.summary(),
            }
        }
        fig = viz.plot_drift_summary(summary)
        assert fig is not None
