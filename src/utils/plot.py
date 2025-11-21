import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch


def save_reconstruction_plot(
    x: torch.Tensor,
    recon: torch.Tensor,
    num_series: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """
    Save a simple reconstruction plot for a few feature dimensions
    of the first sample in the batch.
    """
    os.makedirs("outputs/figures", exist_ok=True)
    if save_path is None:
        save_path = os.path.join("outputs", "figures", "reconstruction.png")

    x_np = x[0].numpy()      # (T, D)
    recon_np = recon[0].numpy()

    T, D = x_np.shape
    num_series = min(num_series, D)

    t_axis = np.arange(T)

    plt.figure(figsize=(10, 6))

    for i in range(num_series):
        plt.subplot(num_series, 1, i + 1)
        plt.plot(t_axis, x_np[:, i], label="Original")
        plt.plot(t_axis, recon_np[:, i], label="Reconstruction", linestyle="--")
        if i == 0:
            plt.legend(loc="upper right")
        plt.xlabel("Time step")
        plt.ylabel(f"Feature {i}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
