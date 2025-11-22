from __future__ import annotations

import torch


def geometric_mask(
    x: torch.Tensor,
    time_mask_ratio: float = 0.15,
    feature_mask_ratio: float = 0.1,
) -> torch.Tensor:
    """
    Apply geometric masking: contiguous time block + subset of features.

    Args:
        x: (B, T, D)
    """
    assert x.ndim == 3
    B, T, D = x.shape
    x_masked = x.clone()

    # Time masking: one block per sample
    block_len = max(1, int(T * time_mask_ratio))
    for b in range(B):
        if block_len >= T:
            start = 0
        else:
            start = torch.randint(0, T - block_len + 1, (1,)).item()
        end = start + block_len
        x_masked[b, start:end, :] = 0.0

    # Feature masking: same feature subset for all samples
    num_mask_features = max(1, int(D * feature_mask_ratio))
    feat_idx = torch.randperm(D)[:num_mask_features]
    x_masked[:, :, feat_idx] = 0.0

    return x_masked
