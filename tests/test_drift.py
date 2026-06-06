"""Tests for the synthetic drift generator."""

import numpy as np
import pytest

from driftwatch.data.synthetic_drift import DriftGenerator


class TestDriftGenerator:
    """Tests for the DriftGenerator."""

    def test_generate_reference(self):
        gen = DriftGenerator(n_features=5, n_reference=1000, random_state=42)
        reference = gen.generate_reference()
        assert reference.shape == (1000, 5)
        assert not np.any(np.isnan(reference))

    def test_covariate_shift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        shifted = gen.covariate_shift(n_samples=200, shift_magnitude=2.0)
        assert shifted.shape == (200, 3)

    def test_prior_shift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        shifted = gen.prior_shift(n_samples=200)
        assert shifted.shape == (200, 3)

    def test_gradual_drift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        result = gen.gradual_drift(n_steps=10, n_per_step=50)
        assert result.shape == (500, 3)

    def test_sudden_drift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        before, after = gen.sudden_drift(n_before=100, n_after=100, shift_magnitude=3.0)
        assert before.shape == (100, 3)
        assert after.shape == (100, 3)

    def test_missingness_drift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        reference = gen.generate_reference()
        missing = gen.missingness_drift(reference, missing_rate=0.3)
        assert np.sum(np.isnan(missing)) > 0

    def test_feature_perturbation_drift(self):
        gen = DriftGenerator(n_features=5, random_state=42)
        reference = gen.generate_reference()
        perturbed = gen.feature_perturbation_drift(reference, noise_std=0.5)
        assert perturbed.shape == reference.shape
        # At least some features should be different
        assert not np.allclose(reference, perturbed)

    def test_gaussian_noise_drift(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        reference = gen.generate_reference()
        noisy = gen.gaussian_noise_drift(reference, noise_std=0.5)
        assert noisy.shape == reference.shape

    def test_feature_corruption_drift(self):
        gen = DriftGenerator(n_features=5, random_state=42)
        reference = gen.generate_reference()
        corrupted = gen.feature_corruption_drift(reference, corruption_value=0.0)
        assert corrupted.shape == reference.shape
        # At least some values should be zero
        assert np.any(corrupted == 0.0)

    def test_demo_pipeline(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        batches = gen.generate_demo_pipeline(n_batches=10, batch_size=50, drift_start_batch=5)
        assert len(batches) == 10
        assert all(b.shape == (50, 3) for b in batches)

    def test_reproducibility(self):
        gen1 = DriftGenerator(n_features=3, random_state=42)
        gen2 = DriftGenerator(n_features=3, random_state=42)
        ref1 = gen1.generate_reference()
        ref2 = gen2.generate_reference()
        assert np.allclose(ref1, ref2)

    def test_drift_increases_over_time(self):
        gen = DriftGenerator(n_features=3, random_state=42)
        # Generate reference from same distribution as clean batches (mean ∈ [-1, 1])
        rng = np.random.default_rng(42)
        ref = np.zeros((500, 3))
        for col in range(3):
            ref[:, col] = rng.normal(rng.uniform(-1, 1), rng.uniform(0.5, 1.0), 500)

        from driftwatch.detectors.psi import PSIDetector

        detector = PSIDetector(threshold=0.1, n_bins=5)
        detector.fit(ref)

        # Score reference against itself — should be very low
        clean_score = detector.score(ref[:100])

        # Score heavily drifted data
        drifted = gen.covariate_shift(n_samples=100, shift_magnitude=5.0)
        drifted_score = detector.score(drifted)

        assert drifted_score > clean_score


class TestDataLoader:
    """Tests for the DataLoader."""

    def test_load_synthetic(self):
        from driftwatch.data.loaders import DataLoader
        loader = DataLoader(random_state=42)
        data = loader.load_synthetic_classification(n_samples=500, n_features=5)
        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "feature_names" in data

    def test_stream_batches_no_drift(self):
        from driftwatch.data.loaders import DataLoader
        loader = DataLoader(random_state=42)
        data = loader.load_synthetic_classification(n_samples=500, n_features=5)
        batches = loader.generate_stream_batches(
            data["X_train"], n_batches=5, batch_size=50
        )
        assert len(batches) == 5

    def test_stream_batches_with_drift(self):
        from driftwatch.data.loaders import DataLoader
        loader = DataLoader(random_state=42)
        data = loader.load_synthetic_classification(n_samples=500, n_features=5)
        batches = loader.generate_stream_batches(
            data["X_train"],
            n_batches=10,
            batch_size=50,
            drift_config={
                "start_batch": 5,
                "drift_type": "noise",
                "magnitude": 2.0,
            },
        )
        assert len(batches) == 10

    def test_image_drift_demo(self):
        from driftwatch.data.loaders import DataLoader
        loader = DataLoader(random_state=42)
        result = loader.generate_image_drift_demo(
            image_size=8, n_reference=50, n_batches=5, batch_size=10
        )
        assert "reference_data" in result
        assert "batches" in result
        assert len(result["batches"]) == 5
