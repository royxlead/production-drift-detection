"""HuggingFace integration adapter for ProductionDriftDetection.

Provides utilities for monitoring HuggingFace transformer models with
text data drift detection.

Uses distilbert-base-uncased-finetuned-sst-2-english as the reference
model for text classification demonstration. Runnable on commodity hardware.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from production_drift_detection.monitors.confidence_monitor import ConfidenceMonitor
from production_drift_detection.monitors.stream_monitor import StreamMonitor
from production_drift_detection.utils.logging import get_logger

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HFAdapter:
    """Adapter for monitoring HuggingFace transformer models with ProductionDriftDetection.

    Uses distilbert-base-uncased-finetuned-sst-2-english by default for
    text classification. Designed to be runnable on commodity hardware.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model ID, by default "distilbert-base-uncased-finetuned-sst-2-english".
    stream_monitor : StreamMonitor, optional
        Monitor for data drift.
    confidence_monitor : ConfidenceMonitor, optional
        Monitor for confidence drift.
    device : int, optional
        Device index (-1 for CPU, 0 for GPU), by default -1.
    max_length : int, optional
        Maximum tokenization length, by default 128.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        stream_monitor: Optional[StreamMonitor] = None,
        confidence_monitor: Optional[ConfidenceMonitor] = None,
        device: int = -1,
        max_length: int = 128,
    ):
        if not HF_AVAILABLE:
            raise ImportError(
                "Transformers and PyTorch are required for HFAdapter. "
                "Install with: pip install production_drift_detection[transformers]"
            )

        self.model_name = model_name
        self.stream_monitor = stream_monitor or StreamMonitor()
        self.confidence_monitor = confidence_monitor or ConfidenceMonitor()
        self.device = device
        self.max_length = max_length
        self._logger = get_logger("production_drift_detection.hf_adapter")

        self._logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_length=max_length,
            truncation=True,
        )

    def predict_text(
        self,
        texts: List[str],
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Make predictions on text data and monitor for drift.

        Parameters
        ----------
        texts : list of str
            Input text samples.
        ground_truth : np.ndarray, optional
            Ground truth labels, if available.

        Returns
        -------
        dict
            Predictions, confidence, and monitoring results.
        """
        # Extract features for drift monitoring (use embedding features)
        features = self._extract_features(texts)

        # Monitor data drift on extracted features
        stream_result = self.stream_monitor.process_batch(features)

        # Get predictions
        results = self._sentiment_pipeline(texts)

        # Extract probabilities (handle positive/negative outputs)
        n_classes = 2
        probabilities = np.zeros((len(results), n_classes))
        predictions = np.zeros(len(results), dtype=int)

        for i, r in enumerate(results):
            score = r["score"]
            if r["label"].upper() == "POSITIVE":
                probabilities[i, 1] = score
                probabilities[i, 0] = 1 - score
                predictions[i] = 1 if score > 0.5 else 0
            else:
                probabilities[i, 0] = score
                probabilities[i, 1] = 1 - score
                predictions[i] = 0 if score > 0.5 else 1

        # Monitor confidence
        conf_result = self.confidence_monitor.update(probabilities, ground_truth)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "raw_outputs": results,
            "drift": stream_result,
            "confidence": conf_result,
        }

    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract feature representations from text for drift monitoring.

        Uses token count and basic statistics as lightweight features.

        Parameters
        ----------
        texts : list of str
            Input texts.

        Returns
        -------
        np.ndarray
            Feature vectors of shape (n_texts, n_features).
        """
        features = []
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="np",
            )
            # Use attention mask and input_ids statistics as features
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            feature_vec = np.array([
                np.mean(input_ids),
                np.std(input_ids),
                np.sum(attention_mask),
                np.sum(input_ids > self.tokenizer.vocab_size // 2),
                len(text.split()),
                np.mean([ord(c) for c in text[:min(len(text), 100)]]),
            ])
            features.append(feature_vec)

        return np.array(features)

    def summary(self) -> Dict[str, Any]:
        """Get combined summary.

        Returns
        -------
        dict
            Summary from all components.
        """
        return {
            "model": self.model_name,
            "device": self.device,
            "stream_monitor": self.stream_monitor.summary(),
            "confidence_monitor": self.confidence_monitor.summary(),
        }
