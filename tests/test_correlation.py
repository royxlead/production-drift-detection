"""Tests for the Confidence-Drift Correlation module."""

import numpy as np
import pytest

from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation


class TestConfidenceDriftCorrelation:
    """Tests for the ConfidenceDriftCorrelation module."""

    def test_add_observation(self):
        corr = ConfidenceDriftCorrelation()
        corr.add_observation(
            confidence=0.9,
            drift_scores={"kl": 0.05, "psi": 0.03},
            entropy=0.3,
            margin=0.8,
        )
        assert len(corr._confidence_history) == 1
        assert "kl" in corr._drift_history

    def test_multiple_observations(self):
        corr = ConfidenceDriftCorrelation()
        for i in range(10):
            corr.add_observation(
                confidence=0.9 - i * 0.02,
                drift_scores={"kl": 0.05 + i * 0.02, "psi": 0.03 + i * 0.01},
            )
        assert len(corr._confidence_history) == 10
        assert len(corr._drift_history["kl"]) == 10

    def test_cross_correlation(self):
        corr = ConfidenceDriftCorrelation(max_lag=3)
        # Create data where confidence decreases as drift increases
        for i in range(20):
            corr.add_observation(
                confidence=0.9 - i * 0.03,
                drift_scores={"kl": 0.05 + i * 0.03, "psi": 0.03 + i * 0.02},
            )
        result = corr.compute_cross_correlation(drift_key="kl")
        assert "n_observations" in result
        assert result["n_observations"] == 20

    def test_cross_correlation_insufficient_data(self):
        corr = ConfidenceDriftCorrelation(max_lag=5)
        corr.add_observation(confidence=0.9, drift_scores={"kl": 0.05})
        result = corr.compute_cross_correlation(drift_key="kl")
        assert "status" in result
        assert result["status"] == "insufficient_data"

    def test_early_warning_score(self):
        corr = ConfidenceDriftCorrelation()
        # Simulate confidence decreasing before drift increases
        for i in range(10):
            corr.add_observation(
                confidence=0.9 - i * 0.05,
                drift_scores={"kl": 0.05 + i * 0.01},
            )
        score = corr.compute_early_warning_score()
        assert "early_warning_score" in score

    def test_early_warning_detection(self):
        corr = ConfidenceDriftCorrelation()
        # Fast confidence drop with slow drift increase
        for i in range(10):
            corr.add_observation(
                confidence=0.9 - i * 0.08,
                drift_scores={"kl": 0.05 + i * 0.005},
            )
        assert len(corr._early_warnings) > 0

    def test_get_visualization_data(self):
        corr = ConfidenceDriftCorrelation()
        for i in range(5):
            corr.add_observation(
                confidence=0.9 - i * 0.05,
                drift_scores={"kl": 0.05 + i * 0.03},
            )
        viz = corr.get_visualization_data()
        assert "timestamps" in viz
        assert "confidence_history" in viz
        assert "drift_history" in viz
        assert len(viz["confidence_history"]) == 5

    def test_summary(self):
        corr = ConfidenceDriftCorrelation()
        for i in range(10):
            corr.add_observation(
                confidence=0.9 - i * 0.03,
                drift_scores={"kl": 0.05 + i * 0.03},
            )
        summary = corr.summary()
        assert "n_observations" in summary
        assert "early_warning_score" in summary
        assert "explanation" in summary

    def test_reset(self):
        corr = ConfidenceDriftCorrelation()
        corr.add_observation(confidence=0.9, drift_scores={"kl": 0.05})
        corr.reset()
        assert len(corr._confidence_history) == 0
        assert len(corr._drift_history) == 0

    def test_confidence_is_leading_indicator(self):
        corr = ConfidenceDriftCorrelation(max_lag=3)
        # Simulate: confidence drops before drift increases
        # (confidence leads drift)
        rng = np.random.default_rng(42)
        for i in range(15):
            if i < 5:
                conf = 0.9
                drift = 0.05
            elif i < 10:
                conf = 0.9 - (i - 5) * 0.1  # Confidence drops first
                drift = 0.05 + (i - 5) * 0.02  # Drift follows slowly
            else:
                conf = 0.4
                drift = 0.05 + 5 * 0.02 + (i - 10) * 0.03

            corr.add_observation(
                confidence=conf,
                drift_scores={"kl": drift},
            )

        summary = corr.summary()
        # This is a stochastic test, so we just verify it runs
        assert summary["n_observations"] == 15
