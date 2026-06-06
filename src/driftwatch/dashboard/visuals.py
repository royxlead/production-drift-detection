"""Visualization utilities for the DriftWatch dashboard.

Uses Plotly for interactive chart generation.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from driftwatch.correlation.confidence_drift import ConfidenceDriftCorrelation
from driftwatch.monitors.confidence_monitor import ConfidenceMonitor


class Visualizer:
    """Generate interactive Plotly visualizations for drift monitoring.

    Parameters
    ----------
    template : str, optional
        Plotly template, by default "plotly_white".
    """

    def __init__(self, template: str = "plotly_white"):
        self.template = template

    def plot_drift_scores(
        self,
        history: Dict[str, List[float]],
        batch_labels: Optional[List[int]] = None,
        title: str = "Drift Scores Over Time",
    ) -> go.Figure:
        """Plot drift scores for multiple detectors.

        Parameters
        ----------
        history : dict of str to list of float
            Detector score histories.
        batch_labels : list of int, optional
            Batch indices for x-axis.
        title : str, optional
            Chart title.

        Returns
        -------
        go.Figure
            Plotly figure.
        """
        if batch_labels is None:
            batch_labels = list(range(1, max(len(v) for v in history.values()) + 1))

        fig = go.Figure()

        colors = {"kl": "#FF6B6B", "psi": "#4ECDC4", "mmd": "#45B7D1", "adwin": "#96CEB4"}

        for det_name, scores in history.items():
            color = colors.get(det_name, None)
            fig.add_trace(go.Scatter(
                x=batch_labels[:len(scores)],
                y=scores,
                mode="lines+markers",
                name=det_name.upper(),
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{det_name.upper()}</b><br>Batch: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Batch",
            yaxis_title="Drift Score",
            template=self.template,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def plot_feature_drift_heatmap(
        self,
        feature_scores: List[Dict[str, float]],
        feature_names: List[str],
    ) -> go.Figure:
        """Plot per-feature drift scores as a heatmap.

        Parameters
        ----------
        feature_scores : list of dict
            Per-feature drift scores for each batch.
        feature_names : list of str
            Feature names.

        Returns
        -------
        go.Figure
            Heatmap figure.
        """
        if not feature_scores:
            return go.Figure()

        data = []
        for batch_scores in feature_scores:
            data.append([batch_scores.get(f, 0) for f in feature_names])

        z_data = np.array(data).T

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=list(range(1, len(feature_scores) + 1)),
            y=feature_names,
            colorscale="RdBu_r",
            zmin=0,
            zmid=None,
            colorbar=dict(title="PSI Score"),
            hovertemplate="Feature: %{y}<br>Batch: %{x}<br>PSI: %{z:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="Per-Feature Drift (PSI)", x=0.5),
            xaxis_title="Batch",
            yaxis_title="Feature",
            template=self.template,
            height=max(300, len(feature_names) * 30),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def plot_confidence_history(
        self,
        confidence_monitor: ConfidenceMonitor,
    ) -> go.Figure:
        """Plot confidence, entropy, and margin history.

        Parameters
        ----------
        confidence_monitor : ConfidenceMonitor
            Confidence monitor with history data.

        Returns
        -------
        go.Figure
            Multi-trace confidence figure.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Confidence", "Entropy", "Margin"),
            vertical_spacing=0.08,
        )

        batches = list(range(1, len(confidence_monitor._confidence_history) + 1))

        fig.add_trace(go.Scatter(
            x=batches,
            y=confidence_monitor._confidence_history,
            mode="lines+markers",
            name="Confidence",
            line=dict(color="#2ECC71", width=2),
            marker=dict(size=6),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=batches,
            y=confidence_monitor._entropy_history,
            mode="lines+markers",
            name="Entropy",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=6),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=batches,
            y=confidence_monitor._margin_history,
            mode="lines+markers",
            name="Margin",
            line=dict(color="#3498DB", width=2),
            marker=dict(size=6),
        ), row=3, col=1)

        fig.update_layout(
            title=dict(text="Confidence Monitoring History", x=0.5),
            template=self.template,
            height=600,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        fig.update_yaxes(title_text="Confidence", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        fig.update_yaxes(title_text="Margin", row=3, col=1)
        fig.update_xaxes(title_text="Batch", row=3, col=1)

        return fig

    def plot_correlation_analysis(
        self,
        correlation: ConfidenceDriftCorrelation,
    ) -> go.Figure:
        """Plot confidence-drift correlation analysis.

        Parameters
        ----------
        correlation : ConfidenceDriftCorrelation
            Correlation module with observation history.

        Returns
        -------
        go.Figure
            Correlation analysis figure.
        """
        viz_data = correlation.get_visualization_data()
        if viz_data["n_observations"] < 2:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Confidence vs Drift Over Time",
                "Cross-Correlation (Lead-Lag)",
                "Confidence Trend",
                "Early Warning Signals",
            ),
            specs=[[{}, {}], [{}, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        batches = list(range(1, viz_data["n_observations"] + 1))

        # Top-left: Confidence vs Drift
        fig.add_trace(go.Scatter(
            x=batches,
            y=viz_data["confidence_history"],
            mode="lines+markers",
            name="Confidence",
            line=dict(color="#2ECC71", width=2),
            marker=dict(size=6),
        ), row=1, col=1)

        # Add first drift detector traces
        if viz_data["drift_history"]:
            det_name = list(viz_data["drift_history"].keys())[0]
            drift_vals = viz_data["drift_history"][det_name]
            fig.add_trace(go.Scatter(
                x=batches[:len(drift_vals)],
                y=drift_vals,
                mode="lines+markers",
                name=f"{det_name.upper()} Drift",
                yaxis="y2",
                line=dict(color="#E74C3C", width=2),
                marker=dict(size=6),
            ), row=1, col=1)

            # Update y-axis to dual-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Drift Score",
                    overlaying="y",
                    side="right",
                    range=[0, max(drift_vals) * 1.2] if drift_vals else None,
                )
            )

        # Top-right: Cross-correlation
        for det_name in viz_data["drift_history"]:
            cc = correlation.compute_cross_correlation(drift_key=det_name)
            if "lead_lag_analysis" in cc:
                lags = [c["lag"] for c in cc["lead_lag_analysis"]]
                lead_corrs = [c.get("confidence_leads_drift", 0) for c in cc["lead_lag_analysis"]]

                fig.add_trace(go.Bar(
                    x=lags,
                    y=lead_corrs,
                    name=f"{det_name.upper()} Lead",
                    marker_color="#3498DB",
                    hovertemplate="Lag: %{x}<br>Correlation: %{y:.4f}<extra></extra>",
                ), row=1, col=2)

        # Bottom-left: Confidence trend with regression
        conf = np.array(viz_data["confidence_history"])
        fig.add_trace(go.Scatter(
            x=batches,
            y=conf,
            mode="lines+markers",
            name="Confidence",
            line=dict(color="#2ECC71", width=2),
            marker=dict(size=6),
        ), row=2, col=1)

        # Add trend line
        if len(conf) > 1:
            z = np.polyfit(batches, conf, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=batches,
                y=p(batches),
                mode="lines",
                name="Trend",
                line=dict(color="#E74C3C", width=2, dash="dash"),
            ), row=2, col=1)

        # Bottom-right: Early warnings
        warnings = viz_data.get("early_warnings", [])
        if warnings:
            # Find the batch index by matching timestamps (more robust than confidence values)
            warning_batches = []
            warning_scores = []
            for w in warnings:
                # Find the closest matching confidence value position
                conf_val = w.get("confidence", 0)
                if viz_data["confidence_history"]:
                    # Use the last occurrence or position-based index
                    idx = len(viz_data["confidence_history"]) - 1
                    for j, c in enumerate(viz_data["confidence_history"]):
                        if abs(c - conf_val) < 0.01:
                            idx = j
                    warning_batches.append(idx + 1)
                    warning_scores.append(w.get("confidence_drop_pct", 0))

            fig.add_trace(go.Scatter(
                x=warning_batches,
                y=warning_scores,
                mode="markers",
                name="Early Warnings",
                marker=dict(
                    color="#E74C3C",
                    size=12,
                    symbol="triangle-down",
                ),
                hovertemplate="Batch: %{x}<br>Drop: %{y:.1f}%<extra></extra>",
            ), row=2, col=2)

        fig.update_layout(
            title=dict(text="Confidence-Drift Correlation Analysis", x=0.5),
            template=self.template,
            height=700,
            showlegend=True,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def plot_detection_latency(
        self,
        scores: np.ndarray,
        threshold: float,
        drift_start_idx: int,
    ) -> go.Figure:
        """Plot detection latency visualization.

        Parameters
        ----------
        scores : np.ndarray
            Drift scores over time.
        threshold : float
            Detection threshold.
        drift_start_idx : int
            Index where drift started.

        Returns
        -------
        go.Figure
            Latency visualization.
        """
        fig = go.Figure()

        batches = list(range(1, len(scores) + 1))

        # Color segments
        colors = []
        for i in range(len(scores)):
            if i < drift_start_idx:
                colors.append("rgba(46, 204, 113, 0.3)")  # Green = no drift
            else:
                colors.append("rgba(231, 76, 60, 0.3)")  # Red = drift

        fig.add_trace(go.Bar(
            x=batches,
            y=scores,
            marker_color=colors,
            name="Score",
            hovertemplate="Batch: %{x}<br>Score: %{y:.4f}<extra></extra>",
        ))

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text=f"Threshold ({threshold})",
        )

        # Drift start line
        fig.add_vline(
            x=drift_start_idx,
            line_dash="dot",
            line_color="#F39C12",
            annotation_text="Drift Introduced",
        )

        fig.update_layout(
            title=dict(text="Detection Latency Analysis", x=0.5),
            xaxis_title="Batch",
            yaxis_title="Drift Score",
            template=self.template,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def plot_drift_summary(
        self,
        monitor_summary: Dict[str, Any],
    ) -> go.Figure:
        """Create a summary dashboard figure.

        Parameters
        ----------
        monitor_summary : dict
            Stream monitor summary data.

        Returns
        -------
        go.Figure
            Summary gauge chart.
        """
        if "detectors" not in monitor_summary:
            return go.Figure()

        detectors = monitor_summary["detectors"]
        names = list(detectors.keys())
        statuses = [det.get("current_status", "unknown") for det in detectors.values()]
        mean_scores = [det.get("mean_score", 0) or 0 for det in detectors.values()]

        # Map status to color
        color_map = {
            "healthy": "#2ECC71",
            "watch": "#F39C12",
            "warning": "#E67E22",
            "critical": "#E74C3C",
        }
        colors = [color_map.get(s, "#95A5A6") for s in statuses]

        fig = go.Figure()

        # Status indicators
        for i, (name, status, score, color) in enumerate(zip(names, statuses, mean_scores, colors)):
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                number=dict(suffix="", font=dict(color=color)),
                title=dict(text=f"{name.upper()}<br><span style='font-size:0.8em;color:{color}'>({status})</span>"),
                gauge=dict(
                    axis=dict(range=[0, max(score * 3, 1)]),
                    bar=dict(color=color),
                    steps=[
                        dict(range=[0, 0.5], color="rgba(46, 204, 113, 0.2)"),
                        dict(range=[0.5, 1.0], color="rgba(241, 196, 15, 0.2)"),
                        dict(range=[1.0, 2.0], color="rgba(230, 126, 34, 0.2)"),
                        dict(range=[2.0, 10], color="rgba(231, 76, 60, 0.2)"),
                    ],
                ),
                domain=dict(row=0, column=i),
            ))

        fig.update_layout(
            grid=dict(rows=1, columns=len(names)),
            template=self.template,
            height=250,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        return fig
