"""Benchmark framework for drift detection.

Provides a framework for comparing detector performance and creating
compatible interfaces for external tools like Evidently AI and NannyML.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from production_drift_detection.detectors.base import BaseDetector
from production_drift_detection.detectors.kl import KLDivergenceDetector
from production_drift_detection.detectors.mmd import MMDDetector
from production_drift_detection.detectors.psi import PSIDetector
from production_drift_detection.evaluation.metrics import (
    compute_detection_latency,
    compute_false_positive_rate,
    compute_sensitivity_to_drift,
    evaluate_detector,
)
from production_drift_detection.utils.logging import get_logger


class BenchmarkFramework:
    """Framework for running and comparing drift detection benchmarks.

    Provides standardized data generation, detector evaluation, and
    result aggregation. Includes interface stubs for comparison with
    Evidently AI and NannyML.

    Parameters
    ----------
    detectors : list of BaseDetector, optional
        Detectors to benchmark. If None, uses default detectors.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDetector]] = None,
        random_state: Optional[int] = None,
    ):
        self.detectors = detectors or [
            KLDivergenceDetector(threshold=0.1, is_categorical=False, n_bins=20),
            PSIDetector(threshold=0.1, n_bins=10),
            MMDDetector(threshold=0.05),
        ]
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._logger = get_logger("production_drift_detection.benchmarks")
        self._results: Dict[str, Any] = {}

    def _generate_benchmark_data(
        self,
        n_features: int = 5,
        n_reference: int = 1000,
        n_clean_batches: int = 5,
        n_drift_batches: int = 10,
        batch_size: int = 100,
        drift_magnitude: float = 1.0,
        drift_type: str = "covariate",
    ) -> dict:
        """Generate standardized benchmark data.

        Parameters
        ----------
        n_features : int, optional
            Number of features, by default 5.
        n_reference : int, optional
            Reference samples, by default 1000.
        n_clean_batches : int, optional
            Clean batches, by default 5.
        n_drift_batches : int, optional
            Drifted batches, by default 10.
        batch_size : int, optional
            Samples per batch, by default 100.
        drift_magnitude : float, optional
            Drift magnitude, by default 1.0.
        drift_type : str, optional
            Type of drift, by default "covariate".

        Returns
        -------
        dict
            Benchmark data.
        """
        from production_drift_detection.data.synthetic_drift import DriftGenerator

        generator = DriftGenerator(
            n_features=n_features,
            n_reference=n_reference,
            random_state=self.random_state,
        )

        reference = generator.generate_reference()

        clean_batches = []
        for _ in range(n_clean_batches):
            idx = self._rng.choice(n_reference, batch_size)
            clean_batches.append(reference[idx].copy())

        drifted_batches = []
        for i in range(n_drift_batches):
            if drift_type == "covariate":
                batch = generator.covariate_shift(
                    n_samples=batch_size,
                    shift_magnitude=drift_magnitude * (i + 1) / n_drift_batches,
                )
            else:
                idx = self._rng.choice(n_reference, batch_size)
                batch = generator.feature_perturbation_drift(
                    reference[idx], noise_std=drift_magnitude * 0.2
                )
            drifted_batches.append(batch)

        return {
            "reference": reference,
            "clean_batches": clean_batches,
            "drifted_batches": drifted_batches,
            "drift_start_batch": n_clean_batches,
        }

    def run_benchmark(
        self,
        n_features: int = 5,
        n_reference: int = 1000,
        n_clean_batches: int = 5,
        n_drift_batches: int = 10,
        batch_size: int = 100,
        drift_magnitude: float = 1.0,
        drift_type: str = "covariate",
    ) -> pd.DataFrame:
        """Run a benchmark across all detectors.

        Parameters
        ----------
        n_features : int, optional
            Number of features.
        n_reference : int, optional
            Reference samples.
        n_clean_batches : int, optional
            Clean batches.
        n_drift_batches : int, optional
            Drifted batches.
        batch_size : int, optional
            Samples per batch.
        drift_magnitude : float, optional
            Drift magnitude.
        drift_type : str, optional
            Drift type.

        Returns
        -------
        pd.DataFrame
            Benchmark results.
        """
        data = self._generate_benchmark_data(
            n_features=n_features,
            n_reference=n_reference,
            n_clean_batches=n_clean_batches,
            n_drift_batches=n_drift_batches,
            batch_size=batch_size,
            drift_magnitude=drift_magnitude,
            drift_type=drift_type,
        )

        results = []
        for detector in self.detectors:
            evaluation = evaluate_detector(
                detector=detector,
                reference_data=data["reference"],
                clean_batches=data["clean_batches"],
                drifted_batches=data["drifted_batches"],
                drift_start_batch=data["drift_start_batch"],
            )

            results.append({
                "detector": detector.name,
                "threshold": detector.threshold,
                "fpr": evaluation["false_positive_rate"]["false_positive_rate"],
                "detection_latency": evaluation["detection_latency"].get("detection_latency_batches", None),
                "drift_detected": evaluation["detection_latency"].get("drift_detected", False),
                "stability": evaluation["stability"]["stability"],
                "detection_rate": evaluation["detection_latency"].get("detection_rate", 0),
            })

        results_df = pd.DataFrame(results)
        self._results = results_df.to_dict(orient="records")
        return results_df

    def run_sensitivity_analysis(
        self,
        magnitudes: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    ) -> pd.DataFrame:
        """Run sensitivity analysis across detectors and drift magnitudes.

        Parameters
        ----------
        magnitudes : list of float, optional
            Drift magnitudes to test.

        Returns
        -------
        pd.DataFrame
            Sensitivity results.
        """
        from production_drift_detection.data.synthetic_drift import DriftGenerator

        generator = DriftGenerator(n_features=5, random_state=self.random_state)

        all_results = []
        for detector in self.detectors:
            detector.fit(generator.generate_reference()[:500])
            sensitivity = compute_sensitivity_to_drift(
                detector=detector,
                drift_generator=generator,
                magnitudes=magnitudes,
            )

            for i, mag in enumerate(magnitudes):
                all_results.append({
                    "detector": detector.name,
                    "magnitude": mag,
                    "mean_score": sensitivity["mean_scores"][i],
                    "std_score": sensitivity["std_scores"][i],
                })

        return pd.DataFrame(all_results)

    def get_evidently_compatible_interface(self) -> dict:
        """Return a description of how to compare with Evidently AI.

        Evidently AI provides ``DataDriftPreset`` and column-level drift
        detection. This method documents the compatibility mapping.

        Returns
        -------
        dict
            Compatibility documentation.
        """
        return {
            "tool": "Evidently AI",
            "comparable_features": [
                "PSI (Population Stability Index)",
                "KL Divergence",
            ],
            "differences": [
                "ProductionDriftDetection focuses on streaming detection; Evidently processes static datasets",
                "ProductionDriftDetection provides confidence-drift correlation",
            ],
            "integration_note": (
                "To compare results, run the same dataset through both "
                "Evidently's DataDriftPreset and ProductionDriftDetection's PSIDetector "
                "and compare per-feature drift scores."
            ),
        }

    def get_nannyml_compatible_interface(self) -> dict:
        """Return a description of how to compare with NannyML.

        NannyML provides data drift detection with statistical tests.
        This method documents the compatibility mapping.

        Returns
        -------
        dict
            Compatibility documentation.
        """
        return {
            "tool": "NannyML",
            "comparable_features": [
                "Univariate drift detection (chi-squared, KS test)",
                "Multivariate drift detection",
            ],
            "differences": [
                "NannyML focuses on post-deployment monitoring without labels",
                "ProductionDriftDetection adds confidence-drift correlation research module",
                "ProductionDriftDetection is designed for streaming workflows",
            ],
            "integration_note": (
                "To compare results, use NannyML's UnivariateDriftCalculator "
                "and ProductionDriftDetection's PSIDetector on the same dataset. Both support "
                "per-feature drift reporting."
            ),
        }

    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary.

        Returns
        -------
        dict
            Summary of benchmark configuration and results.
        """
        return {
            "n_detectors": len(self.detectors),
            "detectors": [d.name for d in self.detectors],
            "results": self._results,
            "evidently_compatibility": self.get_evidently_compatible_interface(),
            "nannyml_compatibility": self.get_nannyml_compatible_interface(),
        }


def benchmark_detector(
    detector: BaseDetector,
    n_trials: int = 5,
) -> Dict[str, Any]:
    """Quick benchmark of a single detector.

    Parameters
    ----------
    detector : BaseDetector
        Detector to benchmark.
    n_trials : int, optional
        Number of trials, by default 5.

    Returns
    -------
    dict
        Benchmark results.
    """
    framework = BenchmarkFramework(detectors=[detector])
    results = framework.run_benchmark()
    return results.to_dict(orient="records")[0] if len(results) > 0 else {}


def compare_detectors(
    detectors: List[BaseDetector],
    drift_magnitudes: List[float] = [0.5, 1.0, 2.0],
) -> pd.DataFrame:
    """Compare multiple detectors across drift magnitudes.

    Parameters
    ----------
    detectors : list of BaseDetector
        Detectors to compare.
    drift_magnitudes : list of float, optional
        Drift magnitudes.

    Returns
    -------
    pd.DataFrame
        Comparison results.
    """
    all_results = []
    for mag in drift_magnitudes:
        framework = BenchmarkFramework(detectors=detectors)
        results = framework.run_benchmark(drift_magnitude=mag)
        results["drift_magnitude"] = mag
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)
