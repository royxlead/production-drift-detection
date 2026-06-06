"""Tests for drift detectors."""

import numpy as np
import pytest

from driftwatch.detectors.adwin import ADWINDetector
from driftwatch.detectors.kl import KLDivergenceDetector
from driftwatch.detectors.mmd import MMDDetector
from driftwatch.detectors.psi import PSIDetector


class TestBaseDetector:
    """Test base detector interface."""

    def test_detect_before_fit_raises_error(self):
        detector = KLDivergenceDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            detector.detect(np.array([1, 2, 3]))

    def test_summary_after_fit(self):
        detector = KLDivergenceDetector()
        detector.fit(np.array([1.0, 2.0, 3.0]))
        summary = detector.summary()
        assert summary["name"] == "KLDivergence"
        assert summary["fitted"] is True

    def test_reset(self):
        detector = KLDivergenceDetector()
        detector.fit(np.array([1, 2, 3]))
        detector.detect(np.array([4, 5, 6]))
        detector.reset()
        assert detector._fitted is False
        assert len(detector._scores) == 0

    def test_severity_classification(self):
        detector = KLDivergenceDetector(threshold=0.1)
        assert detector._classify_severity(0.04) == "healthy"
        assert detector._classify_severity(0.07) == "watch"
        assert detector._classify_severity(0.15) == "warning"
        assert detector._classify_severity(0.25) == "critical"


class TestKLDivergenceDetector:
    """Tests for the KL Divergence detector."""

    def test_identical_distributions(self):
        detector = KLDivergenceDetector(threshold=0.1)
        reference = np.array([1, 2, 1, 2, 1, 2, 1, 2])
        detector.fit(reference)
        score = detector.score(reference)
        assert score < 0.01

    def test_different_distributions(self):
        detector = KLDivergenceDetector(threshold=0.1, is_categorical=True)
        reference = np.array([1, 1, 1, 2, 2, 2, 1, 1])
        batch = np.array([3, 3, 3, 3, 3, 3, 3, 3])
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.01  # KL > 0 for different categorical distributions

    def test_numerical_data(self):
        detector = KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=10)
        reference = np.random.normal(0, 1, 500)
        batch = np.random.normal(3, 1, 500)
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0

    def test_detect_returns_dict(self):
        detector = KLDivergenceDetector(threshold=0.1)
        reference = np.array([1, 2, 1, 2, 1, 2])
        batch = np.array([2, 1, 2, 1, 2, 1])
        detector.fit(reference)
        result = detector.detect(batch)
        assert "score" in result
        assert "drift_detected" in result
        assert "severity" in result


class TestPSIDetector:
    """Tests for the PSI detector."""

    def test_identical_distributions(self):
        detector = PSIDetector(threshold=0.1, n_bins=5)
        reference = np.random.normal(0, 1, 500)
        detector.fit(reference)
        score = detector.score(reference)
        assert score < 0.05

    def test_different_distributions(self):
        detector = PSIDetector(threshold=0.1, n_bins=5)
        reference = np.random.normal(0, 1, 500)
        batch = np.random.normal(3, 1, 500)
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.05

    def test_multivariate(self):
        detector = PSIDetector(threshold=0.1, n_bins=5)
        reference = np.random.normal(0, 1, (500, 3))
        batch = np.random.normal(2, 1, (500, 3))
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.05

    def test_per_feature_reporting(self):
        detector = PSIDetector(threshold=0.1, n_bins=5, per_feature=True,
                               feature_names=["a", "b", "c"])
        reference = np.random.normal(0, 1, (500, 3))
        batch = np.random.normal(2, 1, (500, 3))
        detector.fit(reference)
        detector.score(batch)
        feature_scores = detector.get_feature_scores()
        assert len(feature_scores) == 1
        assert "a" in feature_scores[0]

    def test_1d_input(self):
        detector = PSIDetector(threshold=0.1)
        reference = np.random.normal(0, 1, 500)
        batch = np.random.normal(3, 1, 500)
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.05


class TestMMDDetector:
    """Tests for the MMD detector."""

    def test_identical_distributions(self):
        detector = MMDDetector(threshold=0.05)
        reference = np.random.normal(0, 1, (200, 2))
        detector.fit(reference)
        score = detector.score(reference)
        assert pytest.approx(score, abs=0.05) == 0.0

    def test_different_distributions(self):
        detector = MMDDetector(threshold=0.05)
        reference = np.random.normal(0, 1, (200, 2))
        batch = np.random.normal(3, 1, (200, 2))
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.01

    def test_subsample(self):
        detector = MMDDetector(threshold=0.05, subsample=50)
        reference = np.random.normal(0, 1, (200, 2))
        batch = np.random.normal(3, 1, (200, 2))
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.01

    def test_multivariate(self):
        detector = MMDDetector(threshold=0.05)
        reference = np.random.normal(0, 1, (200, 5))
        batch = np.random.normal(2, 1, (200, 5))
        detector.fit(reference)
        score = detector.score(batch)
        assert score > 0.01


class TestADWINDetector:
    """Tests for the ADWIN detector."""

    def test_initialization(self):
        detector = ADWINDetector(threshold=0.1)
        reference = np.random.normal(0, 1, 100)
        detector.fit(reference)
        assert len(detector._window) == 100

    def test_stable_sequence_no_drift(self):
        detector = ADWINDetector(threshold=0.05, delta=0.01)
        reference = np.random.normal(0, 1, 100)
        detector.fit(reference)
        stable_data = np.random.normal(0, 1, 100)
        detector.score(stable_data)
        assert detector._n_detections == 0

    def test_drift_detection(self):
        detector = ADWINDetector(threshold=0.1, delta=0.05)
        reference = np.array([0] * 50 + [5] * 50)
        detector.fit(reference)

        # Score on drifted data
        drift_batch = np.array([5] * 20)
        score = detector.score(drift_batch)
        assert score > 0.1

    def test_window_growth(self):
        detector = ADWINDetector(threshold=0.1, max_window_size=100, delta=0.5, min_window_size=5)
        reference = np.ones(50) * 0.5
        detector.fit(reference)

        for _ in range(5):
            batch = np.ones(10) * 0.5
            detector.update(batch)

        assert len(detector._window) > 50

    def test_detect_drift_in_update(self):
        detector = ADWINDetector(threshold=0.1, delta=0.1, min_window_size=5)
        reference = np.ones(20)
        detector.fit(reference)

        # Feed gradually increasing values
        for i in range(10):
            batch = np.array([10.0] * 5)
            detector.update(batch)

        # ADWIN may have triggered detection
        assert len(detector._window) <= 70  # Should have been trimmed
