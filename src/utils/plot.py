from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path | None = None,
    num_features: int = 3,
):
    """
    Plot a few feature dimensions of original vs reconstructed.
    """
    T, D = original.shape
    num_features = min(num_features, D)
    t = np.arange(T)

    plt.figure(figsize=(12, 6))
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(t, original[:, i], label="Original")
        plt.plot(t, reconstructed[:, i], linestyle="--", label="Reconstructed")
        if i == 0:
            plt.legend(loc="upper right")
        plt.ylabel(f"Feat {i}")
    plt.xlabel("Time")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
    else:
        plt.tight_layout()
        plt.show()
