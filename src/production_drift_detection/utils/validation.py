"""Input validation utilities for ProductionDriftDetection."""

from typing import Optional, Union

import numpy as np
import pandas as pd


def validate_array(
    data: Union[np.ndarray, list, pd.Series],
    name: str = "data",
    ndim: Optional[int] = None,
    min_length: int = 1,
    allow_nan: bool = False,
) -> np.ndarray:
    """Validate and convert input to a numpy array.

    Parameters
    ----------
    data : array-like
        Input data to validate.
    name : str, optional
        Name for error messages, by default "data".
    ndim : int, optional
        Required number of dimensions, by default None (any).
    min_length : int, optional
        Minimum length along first axis, by default 1.
    allow_nan : bool, optional
        Whether NaN values are allowed, by default False.

    Returns
    -------
    np.ndarray
        Validated numpy array.

    Raises
    ------
    ValueError
        If validation fails.
    TypeError
        If type is not array-like.
    """
    if isinstance(data, pd.Series):
        data = data.values
    if isinstance(data, list):
        data = np.array(data)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name} must be array-like, got {type(data).__name__}")
    if data.size == 0:
        raise ValueError(f"{name} must not be empty")
    if data.ndim == 0:
        data = data.reshape(1)
    if ndim is not None and data.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {data.ndim}")
    if len(data) < min_length:
        raise ValueError(f"{name} must have at least {min_length} samples, got {len(data)}")
    if not allow_nan and np.any(np.isnan(data)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError(f"{name} contains infinite values")
    return data


def validate_dataframe(
    df: pd.DataFrame,
    name: str = "dataframe",
    min_rows: int = 1,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Validate a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    name : str, optional
        Name for error messages, by default "dataframe".
    min_rows : int, optional
        Minimum number of rows, by default 1.
    allow_missing : bool, optional
        Whether missing values are allowed, by default False.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    ValueError
        If validation fails.
    TypeError
        If not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    if len(df) < min_rows:
        raise ValueError(f"{name} must have at least {min_rows} rows, got {len(df)}")
    if df.columns.empty:
        raise ValueError(f"{name} has no columns")
    if not allow_missing and df.isnull().any().any():
        raise ValueError(f"{name} contains missing values")
    return df


def validate_probabilities(
    probs: Union[np.ndarray, list],
    name: str = "probabilities",
    allow_multi_class: bool = True,
) -> np.ndarray:
    """Validate probability values are in [0, 1] and sum to 1 (per row).

    Parameters
    ----------
    probs : array-like
        Probability values.
    name : str, optional
        Name for error messages, by default "probabilities".
    allow_multi_class : bool, optional
        Whether multi-class probabilities are allowed, by default True.

    Returns
    -------
    np.ndarray
        Validated probability array.
    """
    probs = validate_array(probs, name=name, ndim=2 if allow_multi_class else None, allow_nan=False)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"{name} must be in [0, 1]")
    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"{name} rows must sum to 1")
    return probs
