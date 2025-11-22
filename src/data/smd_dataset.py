from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resolve_file(base_dir: Path, subdir: str, machine_id: str) -> Path:
    """
    Try .txt then .csv for flexibility.
    """
    base_dir = Path(base_dir)
    txt = base_dir / subdir / f"{machine_id}.txt"
    csv = base_dir / subdir / f"{machine_id}.csv"

    if txt.exists():
        return txt
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Neither {txt} nor {csv} found.")


def load_smd_machine(
    root_dir: Path,
    machine_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMD train, test, test_label for a single machine.

    Expected naming:
        train/machine-1-1.txt
        test/machine-1-1.txt
        test_label/machine-1-1.txt
    """
    root_dir = Path(root_dir)

    train_path = _resolve_file(root_dir, "train", machine_id)
    test_path = _resolve_file(root_dir, "test", machine_id)
    label_path = _resolve_file(root_dir, "test_label", machine_id)

    train_df = pd.read_csv(train_path, header=None, sep=r"\s+|,|;", engine="python")
    test_df = pd.read_csv(test_path, header=None, sep=r"\s+|,|;", engine="python")
    label_df = pd.read_csv(label_path, header=None, sep=r"\s+|,|;", engine="python")

    train_data = train_df.values.astype(np.float32)
    test_data = test_df.values.astype(np.float32)
    labels = label_df.values.astype(np.int64).squeeze(-1)

    return train_data, test_data, labels


class SlidingWindowDataset(Dataset):
    """
    Converts a long multivariate time series into overlapping windows.

    For train/val, labels are not used but still created as zeros for
    compatibility with DataLoader.
    """

    def __init__(
        self,
        series: np.ndarray,
        labels: np.ndarray | None,
        window_size: int,
        stride: int,
    ) -> None:
        """
        Args:
            series: (T, D)
            labels: (T,) or None
        """
        assert series.ndim == 2, "series must be (T, D)"
        self.series = series
        T = series.shape[0]

        if labels is None:
            self.labels = np.zeros((T,), dtype=np.int64)
        else:
            labels = labels.astype(np.int64).squeeze()
            assert labels.shape[0] == T, "Labels length must match series length"
            self.labels = labels

        self.window_size = window_size
        self.stride = stride
        self.indices = self._make_indices(T, window_size, stride)

    @staticmethod
    def _make_indices(T: int, window_size: int, stride: int):
        idx = []
        start = 0
        while start + window_size <= T:
            idx.append((start, start + window_size))
            start += stride
        return idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x = self.series[start:end]          # (window, D)
        y = self.labels[start:end]          # (window,)

        x_tensor = torch.from_numpy(x)      # float32
        y_tensor = torch.from_numpy(y)      # int64

        return x_tensor, y_tensor
