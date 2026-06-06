"""Synthetic drift generator for creating controlled distribution shifts.

Supports multiple drift types for reproducible experiments and demonstrations.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from driftwatch.utils.logging import get_logger


class DriftGenerator:
    """Generate synthetic data with controlled drift patterns.

    Parameters
    ----------
    n_features : int, optional
        Number of features to generate, by default 5.
    n_reference : int, optional
        Number of reference samples, by default 1000.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int = 5,
        n_reference: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.n_reference = n_reference
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._logger = get_logger("driftwatch.drift_generator")

    def generate_reference(self) -> np.ndarray:
        """Generate a clean reference (training) distribution.

        Returns
        -------
        np.ndarray
            Reference data of shape (n_reference, n_features).
        """
        reference = np.zeros((self.n_reference, self.n_features))
        for col in range(self.n_features):
            mean = self._rng.uniform(-2, 2)
            std = self._rng.uniform(0.5, 1.5)
            reference[:, col] = self._rng.normal(mean, std, self.n_reference)
        return reference

    def generate_mixed_reference(
        self,
        cat_features: int = 2,
        n_categories: int = 3,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate reference data with mixed numerical and categorical features.

        Parameters
        ----------
        cat_features : int, optional
            Number of categorical features, by default 2.
        n_categories : int, optional
            Number of categories per feature, by default 3.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            Numerical and combined reference data.
        """
        numerical = self.generate_reference()
        combined = pd.DataFrame(numerical, columns=[f"num_{i}" for i in range(self.n_features)])

        for i in range(cat_features):
            categories = [f"cat_{j}" for j in range(n_categories)]
            probs = self._rng.dirichlet(np.ones(n_categories))
            combined[f"cat_{i}"] = self._rng.choice(categories, size=self.n_reference, p=probs)

        return numerical, combined

    def covariate_shift(
        self,
        n_samples: int = 500,
        shift_magnitude: float = 1.0,
    ) -> np.ndarray:
        """Generate covariate shift by shifting the mean of features.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate, by default 500.
        shift_magnitude : float, optional
            Magnitude of mean shift, by default 1.0.

        Returns
        -------
        np.ndarray
            Shifted data.
        """
        data = np.zeros((n_samples, self.n_features))
        for col in range(self.n_features):
            mean = self._rng.uniform(-2, 2) + shift_magnitude * self._rng.choice([-1, 1])
            std = self._rng.uniform(0.5, 1.5)
            data[:, col] = self._rng.normal(mean, std, n_samples)
        return data

    def prior_shift(
        self,
        n_samples: int = 500,
        class_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate prior shift by changing class proportions.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples, by default 500.
        class_probs : np.ndarray, optional
            If None, simulates class imbalance shift, by default None.

        Returns
        -------
        np.ndarray
            Data with shifted class balance.
        """
        data = np.zeros((n_samples, self.n_features))
        for col in range(self.n_features):
            mean = self._rng.uniform(-1, 1)
            std = self._rng.uniform(0.5, 1.0)
            data[:, col] = self._rng.normal(mean, std, n_samples)

        # Add a class-separating feature
        if class_probs is None:
            class_probs = np.array([0.8, 0.2])  # Imbalanced
        classes = self._rng.choice(len(class_probs), size=n_samples, p=class_probs)
        data[:, 0] += classes * 2.0  # Separate classes
        return data

    def gradual_drift(
        self,
        n_steps: int = 10,
        n_per_step: int = 100,
        start_magnitude: float = 0.0,
        end_magnitude: float = 3.0,
    ) -> np.ndarray:
        """Generate gradual drift that increases over time.

        Parameters
        ----------
        n_steps : int, optional
            Number of drift steps, by default 10.
        n_per_step : int, optional
            Samples per step, by default 100.
        start_magnitude : float, optional
            Starting drift magnitude, by default 0.0.
        end_magnitude : float, optional
            Ending drift magnitude, by default 3.0.

        Returns
        -------
        np.ndarray
            Data of shape (n_steps * n_per_step, n_features).
        """
        total = n_steps * n_per_step
        data = np.zeros((total, self.n_features))
        magnitudes = np.linspace(start_magnitude, end_magnitude, n_steps)

        for step_idx in range(n_steps):
            mag = magnitudes[step_idx]
            start_idx = step_idx * n_per_step
            end_idx = (step_idx + 1) * n_per_step

            for col in range(self.n_features):
                mean = self._rng.uniform(-2, 2) + mag * self._rng.choice([-1, 1])
                std = self._rng.uniform(0.5, 1.5)
                data[start_idx:end_idx, col] = self._rng.normal(mean, std, n_per_step)

        return data

    def sudden_drift(
        self,
        n_before: int = 500,
        n_after: int = 500,
        shift_magnitude: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a sudden distribution shift at a point.

        Parameters
        ----------
        n_before : int, optional
            Samples before shift, by default 500.
        n_after : int, optional
            Samples after shift, by default 500.
        shift_magnitude : float, optional
            Magnitude of the sudden shift, by default 2.0.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Data before and after the sudden shift.
        """
        before = self.generate_reference()[:n_before]
        after = self.covariate_shift(n_samples=n_after, shift_magnitude=shift_magnitude)
        return before, after

    def missingness_drift(
        self,
        reference: np.ndarray,
        missing_rate: float = 0.3,
    ) -> np.ndarray:
        """Introduce missing values at a given rate.

        Parameters
        ----------
        reference : np.ndarray
            Original data.
        missing_rate : float, optional
            Fraction of values to replace with NaN, by default 0.3.

        Returns
        -------
        np.ndarray
            Data with injected missing values.
        """
        data = reference.copy()
        mask = self._rng.random(data.shape) < missing_rate
        data[mask] = np.nan
        return data

    def feature_perturbation_drift(
        self,
        reference: np.ndarray,
        noise_std: float = 0.5,
        n_corrupt_features: Optional[int] = None,
    ) -> np.ndarray:
        """Add noise to a subset of features.

        Parameters
        ----------
        reference : np.ndarray
            Original data.
        noise_std : float, optional
            Standard deviation of added noise, by default 0.5.
        n_corrupt_features : int, optional
            Number of features to perturb, by default half of features.

        Returns
        -------
        np.ndarray
            Perturbed data.
        """
        data = reference.copy()
        if n_corrupt_features is None:
            n_corrupt_features = max(1, self.n_features // 2)

        corrupt_features = self._rng.choice(
            self.n_features, size=n_corrupt_features, replace=False
        )
        for col in corrupt_features:
            noise = self._rng.normal(0, noise_std, len(data))
            data[:, col] += noise
        return data

    def gaussian_noise_drift(
        self,
        reference: np.ndarray,
        noise_std: float = 0.3,
    ) -> np.ndarray:
        """Add Gaussian noise to all features.

        Parameters
        ----------
        reference : np.ndarray
            Original data.
        noise_std : float, optional
            Standard deviation of noise, by default 0.3.

        Returns
        -------
        np.ndarray
            Noisy data.
        """
        noise = self._rng.normal(0, noise_std, reference.shape)
        return reference + noise

    def feature_corruption_drift(
        self,
        reference: np.ndarray,
        corruption_value: float = 0.0,
        n_corrupt_features: Optional[int] = None,
    ) -> np.ndarray:
        """Set a subset of features to a constant value.

        Parameters
        ----------
        reference : np.ndarray
            Original data.
        corruption_value : float, optional
            Value to set corrupted features to, by default 0.0.
        n_corrupt_features : int, optional
            Number of features to corrupt.

        Returns
        -------
        np.ndarray
            Corrupted data.
        """
        data = reference.copy()
        if n_corrupt_features is None:
            n_corrupt_features = max(1, self.n_features // 3)

        corrupt_features = self._rng.choice(
            self.n_features, size=n_corrupt_features, replace=False
        )
        for col in corrupt_features:
            data[:, col] = corruption_value
        return data

    def generate_demo_pipeline(
        self,
        n_batches: int = 20,
        batch_size: int = 100,
        drift_start_batch: int = 8,
    ) -> list:
        """Generate a complete demo pipeline with controlled drift injection.

        Parameters
        ----------
        n_batches : int, optional
            Number of batches to generate, by default 20.
        batch_size : int, optional
            Samples per batch, by default 100.
        drift_start_batch : int, optional
            Batch index to start injecting drift, by default 8.

        Returns
        -------
        list of np.ndarray
            List of data batches with drift after ``drift_start_batch``.
        """
        batches = []
        drift_magnitude = 0.0

        for batch_idx in range(n_batches):
            if batch_idx < drift_start_batch:
                # Clean data
                batch = np.zeros((batch_size, self.n_features))
                for col in range(self.n_features):
                    mean = self._rng.uniform(-1, 1)
                    std = self._rng.uniform(0.5, 1.0)
                    batch[:, col] = self._rng.normal(mean, std, batch_size)
            else:
                # Increasing drift
                drift_magnitude += 0.15
                batch = self.covariate_shift(
                    n_samples=batch_size, shift_magnitude=drift_magnitude
                )
            batches.append(batch)

        return batches
