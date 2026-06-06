"""PyTorch integration adapter for DriftWatch.

Provides utilities for wrapping PyTorch models with drift monitoring.
Runs with CPU by default for broad compatibility.
"""

from typing import Any, Dict, Optional

import numpy as np

from driftwatch.monitors.confidence_monitor import ConfidenceMonitor
from driftwatch.monitors.stream_monitor import StreamMonitor

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchAdapter:
    """Adapter for monitoring PyTorch models with DriftWatch.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted PyTorch model in evaluation mode.
    stream_monitor : StreamMonitor, optional
        Monitor for data drift.
    confidence_monitor : ConfidenceMonitor, optional
        Monitor for confidence drift.
    device : str, optional
        Device to run inference on, by default "cpu".
    """

    def __init__(
        self,
        model: Optional["nn.Module"] = None,
        stream_monitor: Optional[StreamMonitor] = None,
        confidence_monitor: Optional[ConfidenceMonitor] = None,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PyTorchAdapter. "
                "Install with: pip install driftwatch[pytorch]"
            )

        self.model = model
        self.stream_monitor = stream_monitor or StreamMonitor()
        self.confidence_monitor = confidence_monitor or ConfidenceMonitor()
        self.device = torch.device(device)
        self._softmax = nn.Softmax(dim=1)

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def create_feedforward_classifier(
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ) -> "nn.Module":
        """Create a simple feedforward classifier.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int, optional
            Hidden layer size, by default 64.
        num_classes : int, optional
            Number of output classes, by default 2.

        Returns
        -------
        nn.Module
            Feedforward neural network.
        """
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def predict(self, X: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Make predictions and monitor for drift.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        ground_truth : np.ndarray, optional
            Ground truth labels, if available.

        Returns
        -------
        dict
            Predictions and monitoring results.
        """
        # Monitor data drift
        stream_result = self.stream_monitor.process_batch(X)

        # Get predictions
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).float().to(self.device)
            logits = self.model(tensor_x)
            probabilities = self._softmax(logits).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

        # Monitor confidence
        conf_result = self.confidence_monitor.update(probabilities, ground_truth)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "drift": stream_result,
            "confidence": conf_result,
        }

    def summary(self) -> Dict[str, Any]:
        """Get combined summary.

        Returns
        -------
        dict
            Summary from all monitors.
        """
        return {
            "model": type(self.model).__name__ if self.model else "None",
            "device": str(self.device),
            "stream_monitor": self.stream_monitor.summary(),
            "confidence_monitor": self.confidence_monitor.summary(),
        }
