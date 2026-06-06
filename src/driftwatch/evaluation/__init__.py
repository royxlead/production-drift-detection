"""Evaluation package."""

from driftwatch.evaluation.metrics import (
    compute_detection_latency,
    compute_false_positive_rate,
    compute_detection_stability,
    compute_sensitivity_to_drift,
    evaluate_detector,
)
from driftwatch.evaluation.benchmarks import (
    BenchmarkFramework,
    benchmark_detector,
    compare_detectors,
)

__all__ = [
    "compute_detection_latency",
    "compute_false_positive_rate",
    "compute_detection_stability",
    "compute_sensitivity_to_drift",
    "evaluate_detector",
    "BenchmarkFramework",
    "benchmark_detector",
    "compare_detectors",
]
