"""Confidence-Drift Correlation Module.

This is ProductionDriftDetection's novel research component. It tracks the relationship
between model confidence degradation and distribution drift over time,
providing early warning signals by determining whether confidence changes
precede drift detection.

The module implements:
- Confidence-drift cross-correlation analysis
- Lead-lag relationship estimation
- Early warning indicator generation
- Trend statistics and visualization data
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor
from production_drift_detection.utils.logging import get_logger
from production_drift_detection.utils.stats import compute_entropy


class ConfidenceDriftCorrelation:
    """Analyze the relationship between model confidence and data drift.

    This module addresses the research hypothesis that confidence degradation
    can serve as an earlier warning signal than accuracy degradation or
    drift scores alone.

    Parameters
    ----------
    max_lag : int, optional
        Maximum lag for cross-correlation analysis, by default 10.
    name : str, optional
        Module name.
    """

    def __init__(self, max_lag: int = 10, name: Optional[str] = None):
        self.max_lag = max_lag
        self.name = name or "ConfidenceDriftCorrelation"

        self._confidence_history: List[float] = []
        self._drift_history: Dict[str, List[float]] = {}
        self._timestamps: List[pd.Timestamp] = []
        self._early_warnings: List[Dict[str, Any]] = []

        self._logger = get_logger(f"production_drift_detection.{self.name}")

    def add_observation(
        self,
        confidence: float,
        drift_scores: Dict[str, float],
        entropy: Optional[float] = None,
        margin: Optional[float] = None,
    ) -> None:
        """Add a single observation point.

        Parameters
        ----------
        confidence : float
            Mean confidence for the current batch.
        drift_scores : dict of str to float
            Drift scores from each detector.
        entropy : float, optional
            Mean entropy for the current batch.
        margin : float, optional
            Mean margin for the current batch.
        """
        self._confidence_history.append(confidence)
        self._timestamps.append(pd.Timestamp.now())

        for det_name, score in drift_scores.items():
            if det_name not in self._drift_history:
                self._drift_history[det_name] = []
            self._drift_history[det_name].append(score)

        # Check for early warning
        warning = self._check_early_warning(confidence, drift_scores)
        if warning:
            self._early_warnings.append(warning)

    def _check_early_warning(
        self,
        confidence: float,
        drift_scores: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Check if this observation forms an early warning signal.

        An early warning is flagged when confidence drops significantly
        while drift scores are still low, suggesting confidence may be
        a leading indicator.

        Parameters
        ----------
        confidence : float
            Current confidence.
        drift_scores : dict
            Current drift scores.

        Returns
        -------
        dict or None
            Early warning information if detected.
        """
        if len(self._confidence_history) < 3:
            return None

        # Compute baseline confidence (first 3 observations)
        baseline_conf = np.mean(self._confidence_history[:3])
        conf_drop = (baseline_conf - confidence) / max(baseline_conf, 1e-10)

        # Check if confidence dropped but drift is still low
        mean_drift = np.mean(list(drift_scores.values()))
        thresholds = [0.1, 0.1, 0.05]  # Default thresholds for KL, PSI, MMD
        avg_threshold = np.mean(thresholds)

        if conf_drop > 0.1 and mean_drift < avg_threshold:
            return {
                "timestamp": pd.Timestamp.now(),
                "confidence_drop_pct": float(conf_drop * 100),
                "mean_drift_score": float(mean_drift),
                "confidence": float(confidence),
                "drift_scores": drift_scores,
                "signal": "confidence_degrading_before_drift",
            }

        return None

    def compute_cross_correlation(
        self,
        drift_key: str = "kl",
        confidence_metric: str = "confidence",
    ) -> Dict[str, Any]:
        """Compute cross-correlation between confidence and drift series.

        Parameters
        ----------
        drift_key : str, optional
            Which drift detector to use, by default "kl".
        confidence_metric : str, optional
            Which confidence metric to use, by default "confidence".

        Returns
        -------
        dict
            Cross-correlation analysis results.
        """
        if drift_key not in self._drift_history:
            return {"error": f"No data for drift detector: {drift_key}"}

        conf = np.array(self._confidence_history)
        drift = np.array(self._drift_history[drift_key])

        if len(conf) < self.max_lag + 2:
            return {"status": "insufficient_data", "n_observations": len(conf)}

        # Normalize both series
        conf_norm = (conf - np.mean(conf)) / max(np.std(conf), 1e-10)
        drift_norm = (drift - np.mean(drift)) / max(np.std(drift), 1e-10)

        # Compute cross-correlation for lags
        correlations = []
        for lag in range(0, min(self.max_lag, len(conf) - 1)):
            if lag > 0 and len(conf_norm[lag:]) != len(drift_norm[:-lag]):
                continue
            if lag == 0:
                corr = float(np.corrcoef(conf_norm, drift_norm)[0, 1])
                correlations.append({"lag": lag, "correlation": corr, "direction": "simultaneous"})
            else:
                # Confidence leads drift (confidence at t-lag correlated with drift at t)
                c_lead = conf_norm[:-lag]
                d_lead = drift_norm[lag:]
                corr_lead = 0.0
                if len(c_lead) > 2 and len(d_lead) > 2:
                    corr_lead = float(np.corrcoef(c_lead, d_lead)[0, 1])

                # Drift leads confidence
                c_lag = conf_norm[lag:]
                d_lag = drift_norm[:-lag]
                corr_lag = 0.0
                if len(c_lag) > 2 and len(d_lag) > 2:
                    corr_lag = float(np.corrcoef(c_lag, d_lag)[0, 1])

                correlations.append({"lag": lag, "confidence_leads_drift": corr_lead, "drift_leads_confidence": corr_lag})

        # Determine optimal lead-lag
        lead_corrs = [c.get("confidence_leads_drift", 0) for c in correlations if "confidence_leads_drift" in c]
        lag_corrs = [c.get("drift_leads_confidence", 0) for c in correlations if "drift_leads_confidence" in c]

        best_lead_lag = 0
        simultaneous_corr = abs(correlations[0].get("correlation", 0)) if correlations else 0

        # Check if confidence is a leading indicator
        if lead_corrs:
            max_lead_corr = max(abs(c) for c in lead_corrs)
            max_lag_corr = max(abs(c) for c in lag_corrs) if lag_corrs else 0

            if max_lead_corr > max_lag_corr and max_lead_corr > simultaneous_corr:
                best_idx = np.argmax([abs(c) for c in lead_corrs])
                best_lead_lag = best_idx + 1  # +1 because lag 0 is excluded

        return {
            "n_observations": len(conf),
            "confidence_trend": float(conf[-1] - conf[0]),
            "drift_trend": float(drift[-1] - drift[0]),
            "simultaneous_correlation": correlations[0] if correlations else None,
            "lead_lag_analysis": correlations,
            "confidence_is_leading_indicator": best_lead_lag > 0,
            "optimal_lag": best_lead_lag,
            "drift_std": float(np.std(drift)),
            "confidence_std": float(np.std(conf)),
        }

    def compute_early_warning_score(self) -> Dict[str, Any]:
        """Compute an early warning score based on confidence-drift dynamics.

        The early warning score combines:
        - Confidence degradation rate
        - Drift-to-confidence lag
        - Entropy increase rate (if available)

        Higher scores indicate stronger early warning signals.

        Returns
        -------
        dict
            Early warning score and components.
        """
        if len(self._confidence_history) < 4:
            return {"score": 0.0, "status": "insufficient_data"}

        conf = np.array(self._confidence_history)

        # Confidence degradation rate (negative = dropping)
        first_half = np.mean(conf[: len(conf) // 2])
        second_half = np.mean(conf[len(conf) // 2:])
        degradation_rate = (second_half - first_half) / max(first_half, 1e-10)

        # Drift acceleration
        drift_acceleration = 0.0
        for det_name, drift_vals in self._drift_history.items():
            if len(drift_vals) >= 4:
                d = np.array(drift_vals)
                first_drift = np.mean(d[: len(d) // 2])
                second_drift = np.mean(d[len(d) // 2:])
                accel = (second_drift - first_drift) / max(first_drift, 1e-10)
                drift_acceleration = max(drift_acceleration, accel)

        # Lead-lag advantage
        cross_corr = {}
        for det_name in self._drift_history:
            cross_corr[det_name] = self.compute_cross_correlation(drift_key=det_name)

        confidence_leads = any(
            cc.get("confidence_is_leading_indicator", False) for cc in cross_corr.values()
        )

        # Combine into early warning score (0-100)
        score = 0.0
        components = {}

        if degradation_rate < -0.05:
            conf_component = min(abs(degradation_rate) * 50, 50)
            score += conf_component
            components["confidence_degradation"] = conf_component

        if drift_acceleration > 0.1:
            drift_component = min(drift_acceleration * 30, 30)
            score += drift_component
            components["drift_acceleration"] = drift_component

        if confidence_leads:
            lead_component = 20.0
            score += lead_component
            components["leading_indicator"] = lead_component

        components["early_warning_score"] = min(score, 100)
        components["degradation_rate"] = float(degradation_rate)
        components["drift_acceleration"] = float(drift_acceleration)
        components["confidence_is_leading"] = confidence_leads

        return components

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization.

        Returns
        -------
        dict
            Visualization-ready data with time series and correlation data.
        """
        return {
            "timestamps": [str(t) for t in self._timestamps],
            "confidence_history": self._confidence_history,
            "drift_history": self._drift_history,
            "early_warnings": self._early_warnings,
            "n_observations": len(self._confidence_history),
        }

    def summary(self) -> Dict[str, Any]:
        """Comprehensive summary of the correlation analysis.

        Returns
        -------
        dict
            Summary with all key findings.
        """
        score = self.compute_early_warning_score()
        cross_correlations = {}
        for det_name in self._drift_history:
            cross_correlations[det_name] = self.compute_cross_correlation(drift_key=det_name)

        return {
            "name": self.name,
            "n_observations": len(self._confidence_history),
            "n_early_warnings": len(self._early_warnings),
            "early_warning_score": score.get("early_warning_score", 0),
            "confidence_trend": (
                "decreasing" if self._confidence_history and self._confidence_history[-1] < np.mean(self._confidence_history[:2])
                else "stable"
            ) if len(self._confidence_history) >= 2 else "insufficient",
            "confidence_is_leading_indicator": any(
                cc.get("confidence_is_leading_indicator", False) for cc in cross_correlations.values()
            ),
            "cross_correlations": cross_correlations,
            "explanation": (
                "Confidence-drift correlation measures whether changes in model "
                "confidence can serve as early warning signals for distribution shift. "
                "When confidence degrades before drift scores increase, it suggests "
                "confidence monitoring provides leading indicators for model degradation."
            ),
        }

    def reset(self) -> None:
        """Reset all stored data."""
        self._confidence_history = []
        self._drift_history = {}
        self._timestamps = []
        self._early_warnings = []
        self._logger.info("ConfidenceDriftCorrelation reset")
