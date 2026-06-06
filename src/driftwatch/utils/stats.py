"""Statistical utilities for DriftWatch."""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional, Tuple


def compute_entropy(probabilities: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute entropy of probability distributions.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability values of shape (n_samples, n_classes).
    axis : int, optional
        Axis to compute entropy over, by default 1.

    Returns
    -------
    np.ndarray
        Entropy values of shape (n_samples,).
    """
    p = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def compute_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Compute confidence as the maximum probability per sample.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability values of shape (n_samples, n_classes).

    Returns
    -------
    np.ndarray
        Confidence values of shape (n_samples,).
    """
    return np.max(probabilities, axis=1)


def compute_margin(probabilities: np.ndarray) -> np.ndarray:
    """Compute margin as difference between top two probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability values of shape (n_samples, n_classes).

    Returns
    -------
    np.ndarray
        Margin values of shape (n_samples,).
    """
    if probabilities.shape[1] < 2:
        return np.zeros(probabilities.shape[0])
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def compute_psi_bins(
    reference: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin edges and bin proportions for PSI calculation.

    Parameters
    ----------
    reference : np.ndarray
        Reference (training) data.
    actual : np.ndarray
        Actual (production) data.
    n_bins : int, optional
        Number of bins, by default 10.
    bin_strategy : str, optional
        Binning strategy ("quantile" or "uniform"), by default "quantile".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (bin_edges, reference_proportions, actual_proportions).
    """
    combined = np.concatenate([reference, actual])
    if bin_strategy == "quantile":
        bin_edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    else:
        bin_edges = np.linspace(np.min(combined), np.max(combined), n_bins + 1)

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([np.min(combined) - 0.5, np.max(combined) + 0.5])

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    ref_props = ref_counts / max(len(reference), 1)
    actual_props = actual_counts / max(len(actual), 1)

    return bin_edges, ref_props, actual_props


def smooth_distribution(
    probs: np.ndarray, eps: float = 1e-10
) -> np.ndarray:
    """Apply Laplace smoothing to a probability distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability values.
    eps : float, optional
        Small smoothing constant, by default 1e-10.

    Returns
    -------
    np.ndarray
        Smoothed probability values that sum to 1.
    """
    smoothed = probs + eps
    return smoothed / np.sum(smoothed)


def compute_rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix between X and Y.

    Parameters
    ----------
    X : np.ndarray
        First matrix of shape (n, d).
    Y : np.ndarray
        Second matrix of shape (m, d).
    bandwidth : float, optional
        Kernel bandwidth. If None, use median heuristic.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (n, m).
    """
    if bandwidth is None:
        # Median heuristic for bandwidth selection
        if len(X) > 100 or len(Y) > 100:
            sample = np.vstack([X[: min(100, len(X))], Y[: min(100, len(Y))]])
        else:
            sample = np.vstack([X, Y])
        dists = cdist(sample, sample, metric="euclidean")
        bandwidth = np.median(dists[dists > 0])
        if bandwidth == 0 or np.isnan(bandwidth):
            bandwidth = 1.0

    K_xy = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-K_xy / (2 * bandwidth**2))


def compute_rolling_statistics(
    values: np.ndarray,
    window: int = 10,
) -> dict:
    """Compute rolling statistics over a time series.

    Parameters
    ----------
    values : np.ndarray
        Time series values.
    window : int, optional
        Rolling window size, by default 10.

    Returns
    -------
    dict
        Dictionary with rolling mean, std, min, max.
    """
    if len(values) < window:
        window = len(values)

    result = {}
    result["rolling_mean"] = np.convolve(values, np.ones(window) / window, mode="valid")
    result["rolling_std"] = np.array([
        np.std(values[max(0, i - window + 1): i + 1]) for i in range(window - 1, len(values))
    ])
    result["rolling_min"] = np.array([
        np.min(values[max(0, i - window + 1): i + 1]) for i in range(window - 1, len(values))
    ])
    result["rolling_max"] = np.array([
        np.max(values[max(0, i - window + 1): i + 1]) for i in range(window - 1, len(values))
    ])
    return result
