"""Data loaders for DriftWatch demonstrations.

Provides utilities for loading common datasets and preparing them
for drift monitoring demonstrations.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer, load_iris

from driftwatch.utils.logging import get_logger


class DataLoader:
    """Utility class for loading and preparing datasets for drift demonstrations.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._logger = get_logger("driftwatch.data_loader")

    def load_synthetic_classification(
        self,
        n_samples: int = 2000,
        n_features: int = 10,
        n_informative: int = 5,
        n_classes: int = 2,
        test_size: float = 0.3,
    ) -> dict:
        """Create a synthetic classification dataset.

        Parameters
        ----------
        n_samples : int, optional
            Total samples, by default 2000.
        n_features : int, optional
            Number of features, by default 10.
        n_informative : int, optional
            Informative features, by default 5.
        n_classes : int, optional
            Number of classes, by default 2.
        test_size : float, optional
            Test set proportion, by default 0.3.

        Returns
        -------
        dict
            Dictionary with X_train, X_test, y_train, y_test, and feature_names.
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            random_state=self.random_state,
        )

        split = int(len(X) * (1 - test_size))
        indices = self._rng.permutation(len(X))
        train_idx = indices[:split]
        test_idx = indices[split:]

        feature_names = [f"feature_{i}" for i in range(n_features)]

        return {
            "X_train": X[train_idx],
            "X_test": X[test_idx],
            "y_train": y[train_idx],
            "y_test": y[test_idx],
            "feature_names": feature_names,
            "n_classes": n_classes,
        }

    def load_breast_cancer(self) -> dict:
        """Load the breast cancer dataset.

        Returns
        -------
        dict
            Dataset with X_train, X_test, y_train, y_test, feature_names.
        """
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)

        split = int(len(X) * 0.7)
        indices = self._rng.permutation(len(X))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return {
            "X_train": X[train_idx],
            "X_test": X[test_idx],
            "y_train": y[train_idx],
            "y_test": y[test_idx],
            "feature_names": feature_names,
            "n_classes": 2,
        }

    def load_iris(self) -> dict:
        """Load the iris dataset.

        Returns
        -------
        dict
            Dataset with X_train, X_test, y_train, y_test, feature_names.
        """
        data = load_iris()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)

        split = int(len(X) * 0.7)
        indices = self._rng.permutation(len(X))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return {
            "X_train": X[train_idx],
            "X_test": X[test_idx],
            "y_train": y[train_idx],
            "y_test": y[test_idx],
            "feature_names": feature_names,
            "n_classes": 3,
        }

    def generate_stream_batches(
        self,
        reference_data: np.ndarray,
        n_batches: int = 20,
        batch_size: int = 100,
        drift_config: Optional[dict] = None,
    ) -> list:
        """Generate streaming batches with optional drift.

        Parameters
        ----------
        reference_data : np.ndarray
            Reference data to sample from.
        n_batches : int, optional
            Number of batches, by default 20.
        batch_size : int, optional
            Samples per batch, by default 100.
        drift_config : dict, optional
            Drift configuration with keys:
            - start_batch: batch to start drift
            - drift_type: type of drift
            - magnitude: drift magnitude

        Returns
        -------
        list of np.ndarray
            Streaming data batches.
        """
        from driftwatch.data.synthetic_drift import DriftGenerator

        generator = DriftGenerator(
            n_features=reference_data.shape[1],
            random_state=self.random_state,
        )

        batches = []
        for batch_idx in range(n_batches):
            if drift_config and batch_idx >= drift_config.get("start_batch", 999):
                magnitude = drift_config.get("magnitude", 1.0)
                drift_type = drift_config.get("drift_type", "covariate_shift")
                if drift_type == "covariate_shift":
                    batch = generator.covariate_shift(
                        n_samples=batch_size, shift_magnitude=magnitude * (batch_idx - drift_config["start_batch"] + 1) * 0.2
                    )
                elif drift_type == "noise":
                    ref_sample = reference_data[self._rng.choice(len(reference_data), batch_size)]
                    batch = generator.gaussian_noise_drift(ref_sample, noise_std=magnitude * 0.2)
                elif drift_type == "feature_perturbation":
                    ref_sample = reference_data[self._rng.choice(len(reference_data), batch_size)]
                    batch = generator.feature_perturbation_drift(ref_sample, noise_std=magnitude * 0.2)
                else:
                    batch = generator.covariate_shift(
                        n_samples=batch_size, shift_magnitude=magnitude
                    )
            else:
                idx = self._rng.choice(len(reference_data), batch_size)
                batch = reference_data[idx].copy()

            batches.append(batch)

        return batches

    def generate_image_drift_demo(
        self,
        image_size: int = 32,
        n_reference: int = 500,
        n_batches: int = 10,
        batch_size: int = 50,
    ) -> dict:
        """Generate synthetic image data for drift monitoring demo.

        Uses flattened feature vectors with controlled noise to simulate
        image distribution drift. Works entirely offline.

        Parameters
        ----------
        image_size : int, optional
            Image size (image_size x image_size), by default 32.
        n_reference : int, optional
            Reference samples, by default 500.
        n_batches : int, optional
            Number of batches, by default 10.
        batch_size : int, optional
            Samples per batch, by default 50.

        Returns
        -------
        dict
            Dictionary with reference_data and batches.
        """
        from driftwatch.data.synthetic_drift import DriftGenerator
        n_pixels = image_size * image_size
        generator = DriftGenerator(n_features=n_pixels, random_state=self.random_state)

        # Generate reference "images" as flattened vectors
        reference = generator.generate_reference()[:n_reference]

        # Generate batches with increasing drift
        batches = []
        for batch_idx in range(n_batches):
            magnitude = batch_idx * 0.3
            if magnitude > 0:
                batch = generator.covariate_shift(
                    n_samples=batch_size, shift_magnitude=magnitude
                )
            else:
                idx = self._rng.choice(len(reference), batch_size)
                batch = reference[idx].copy()
            batches.append(batch)

        return {
            "reference_data": reference,
            "batches": batches,
            "image_size": image_size,
        }
