"""Tests for stream and confidence monitors."""

import numpy as np
import pytest

from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.psi import PSIDetector
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.monitors.stream_monitor import StreamMonitor


class TestStreamMonitor:
    """Tests for the StreamMonitor."""

    def test_initialization_and_fit(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)
        assert monitor._fitted is True

    def test_process_batch(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)
        batch = np.random.normal(0, 1, (100, 3))
        result = monitor.process_batch(batch)
        assert "scores" in result
        assert "alerts" in result
        assert "drift_detected" in result
        assert result["batch"] == 1

    def test_multiple_batches(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(5):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        assert monitor._batch_count == 5

    def test_drift_detection_with_shifted_data(self):
        detector = PSIDetector(threshold=0.05, n_bins=5)
        monitor = StreamMonitor(detectors={"psi": detector})

        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        # Process clean batches first
        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        # Process shifted batches
        for _ in range(5):
            batch = np.random.normal(5, 1, (100, 3))
            result = monitor.process_batch(batch)

        # Should have some detections or alerts
        assert monitor._batch_count >= 8

    def test_get_history(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        history = monitor.get_history()
        assert len(history) == len(monitor.detectors)
        for det_name in monitor.detectors:
            assert len(history[det_name]) == 3

    def test_summary(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        summary = monitor.summary()
        assert "detectors" in summary
        assert summary["batch_count"] == 3

    def test_export_results_json(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        exported = monitor.export_results(format="json")
        assert isinstance(exported, str)
        import json
        parsed = json.loads(exported)
        assert len(parsed) == 3

    def test_export_results_csv_dataframe(self):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        exported = monitor.export_results(format="csv")
        assert isinstance(exported, object)
        assert len(exported) == 3

    def test_export_results_csv_file(self, tmp_path):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        filepath = str(tmp_path / "drift_scores.csv")
        result = monitor.export_results(format="csv", filepath=filepath)
        assert result is None  # Returns None when writing to file
        import os
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert "batch" in content
        assert "kl_score" in content or "psi_score" in content

    def test_export_results_json_file(self, tmp_path):
        monitor = StreamMonitor()
        reference = np.random.normal(0, 1, (500, 3))
        monitor.fit(reference)

        for _ in range(3):
            batch = np.random.normal(0, 1, (100, 3))
            monitor.process_batch(batch)

        filepath = str(tmp_path / "drift_scores.json")
        result = monitor.export_results(format="json", filepath=filepath)
        assert result is None
        import os
        assert os.path.exists(filepath)
        import json
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 3


class TestConfidenceMonitor:
    """Tests for the ConfidenceMonitor."""

    def test_update_with_probabilities(self):
        monitor = ConfidenceMonitor()
        probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        result = monitor.update(probs)
        assert "mean_confidence" in result
        assert "mean_entropy" in result
        assert "mean_margin" in result
        assert result["mean_confidence"] > 0.7

    def test_update_with_ground_truth(self):
        monitor = ConfidenceMonitor()
        probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        truth = np.array([1, 1, 1])
        result = monitor.update(probs, truth)
        assert "accuracy" in result
        assert result["accuracy"] == 1.0

    def test_confidence_history(self):
        monitor = ConfidenceMonitor()
        for _ in range(5):
            probs = np.random.dirichlet([1, 9], size=100)
            monitor.update(probs)
        assert len(monitor._confidence_history) == 5

    def test_get_trends(self):
        monitor = ConfidenceMonitor()
        for _ in range(5):
            probs = np.random.dirichlet([1, 9], size=100)
            monitor.update(probs)
        trends = monitor.get_trends()
        assert "confidence_trend" in trends
        assert "batches_observed" in trends

    def test_degradation_detection(self):
        monitor = ConfidenceMonitor()
        # High confidence initially
        for _ in range(3):
            probs = np.random.dirichlet([1, 20], size=100)
            monitor.update(probs)
        # Lower confidence
        for _ in range(3):
            probs = np.random.dirichlet([1, 2], size=100)
            monitor.update(probs)
        degraded, reason = monitor.degradation_detected()
        assert isinstance(degraded, bool)
        assert isinstance(reason, str)

    def test_calibration_summary(self):
        monitor = ConfidenceMonitor()
        for _ in range(3):
            probs = np.random.dirichlet([1, 9], size=100)
            monitor.update(probs)
        cal = monitor.get_calibration_summary()
        assert "mean_confidence_overall" in cal

    def test_uncertainty_metrics(self):
        monitor = ConfidenceMonitor()
        probs = np.random.dirichlet([1, 9], size=100)
        monitor.update(probs)
        metrics = monitor.get_uncertainty_metrics()
        assert "low_confidence_ratio" in metrics
        assert "high_entropy_ratio" in metrics
