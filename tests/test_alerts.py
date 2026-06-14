"""Tests for the alerting system."""

import pytest

from production_drift_detection.alerts.rules import AlertEngine, RollingWindowRule, ThresholdRule
from production_drift_detection.alerts.schemas import Alert, Severity


class TestAlertSchema:
    """Tests for the Alert dataclass."""

    def test_alert_creation(self):
        alert = Alert(
            detector="kl",
            score=0.5,
            threshold=0.1,
            severity=Severity.WARNING,
            explanation="High drift detected",
        )
        assert alert.detector == "kl"
        assert alert.score == 0.5
        assert alert.severity == Severity.WARNING

    def test_alert_to_dict(self):
        alert = Alert(
            detector="psi",
            score=0.3,
            threshold=0.1,
            severity=Severity.CRITICAL,
        )
        d = alert.to_dict()
        assert d["detector"] == "psi"
        assert d["severity"] == "critical"

    def test_alert_from_dict(self):
        d = {
            "detector": "mmd",
            "score": 0.2,
            "threshold": 0.05,
            "timestamp": "2024-01-01T00:00:00",
            "severity": "warning",
            "explanation": "",
            "metadata": {},
        }
        alert = Alert.from_dict(d)
        assert alert.detector == "mmd"
        assert alert.severity == Severity.WARNING

    def test_severity_comparison(self):
        assert Severity.HEALTHY < Severity.WARNING
        assert Severity.WARNING < Severity.CRITICAL
        assert Severity.CRITICAL >= Severity.WARNING
        assert Severity.HEALTHY < Severity.CRITICAL


class TestThresholdRule:
    """Tests for the ThresholdRule."""

    def test_below_threshold(self):
        rule = ThresholdRule()
        alert = rule.evaluate("kl", 0.05, 0.1)
        assert alert is None

    def test_warning_alert(self):
        rule = ThresholdRule()
        alert = rule.evaluate("kl", 0.15, 0.1)
        assert alert is not None
        assert alert.severity == Severity.WARNING

    def test_critical_alert(self):
        rule = ThresholdRule(critical_multiplier=2.0)
        alert = rule.evaluate("kl", 0.25, 0.1)
        assert alert is not None
        assert alert.severity == Severity.CRITICAL


class TestRollingWindowRule:
    """Tests for the RollingWindowRule."""

    def test_insufficient_history(self):
        rule = RollingWindowRule(window_size=5, min_consecutive=3)
        # Only 2 data points
        alert = rule.evaluate("kl", 0.2, 0.1)
        assert alert is None

    def test_consecutive_exceedances(self):
        rule = RollingWindowRule(window_size=5, min_consecutive=3)
        # Add 3 consecutive high scores
        for _ in range(3):
            alert = rule.evaluate("kl", 0.2, 0.1)
        assert alert is not None
        assert alert.severity == Severity.WARNING

    def test_not_enough_consecutive(self):
        rule = RollingWindowRule(window_size=5, min_consecutive=3)
        # Mix of high and low
        rule.evaluate("kl", 0.2, 0.1)
        rule.evaluate("kl", 0.05, 0.1)
        rule.evaluate("kl", 0.2, 0.1)
        assert rule.evaluate("kl", 0.05, 0.1) is None


class TestAlertEngine:
    """Tests for the AlertEngine."""

    def test_multiple_rules(self):
        engine = AlertEngine(rules=[ThresholdRule(), RollingWindowRule()])
        alerts = engine.evaluate("kl", 0.15, 0.1)
        assert len(alerts) >= 1  # Threshold should trigger

    def test_get_alerts(self):
        engine = AlertEngine()
        engine.evaluate("kl", 0.15, 0.1)
        engine.evaluate("psi", 0.2, 0.1)

        alerts = engine.get_alerts()
        assert len(alerts) >= 1

    def test_clear(self):
        engine = AlertEngine()
        engine.evaluate("kl", 0.15, 0.1)
        engine.clear()
        assert len(engine.get_alerts()) == 0

    def test_filter_by_detector(self):
        engine = AlertEngine(rules=[ThresholdRule()])
        engine.evaluate("kl", 0.15, 0.1)
        engine.evaluate("psi", 0.2, 0.1)

        kl_alerts = engine.get_alerts(detector="kl")
        assert all(a.detector == "kl" for a in kl_alerts)
