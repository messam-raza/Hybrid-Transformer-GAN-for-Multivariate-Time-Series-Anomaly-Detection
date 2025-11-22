from __future__ import annotations

import numpy as np


def compute_zscore_params(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std over time for each feature.

    Args:
        train_data: (T, D)

    Returns:
        mean: (1, D)
        std: (1, D)
    """
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def apply_zscore(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization with precomputed mean/std.
    """
    return (data - mean) / std
