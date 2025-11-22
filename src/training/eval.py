from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.smd_dataset import load_smd_machine, SlidingWindowDataset
from src.data.preprocessing import compute_zscore_params, apply_zscore
from src.models.transformer import TimeSeriesTransformerAE
from src.utils.metrics import evaluate_window_metrics
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid Transformer-GAN TSAD on SMD."
    )
    parser.add_argument(
        "--machine_id", type=str, default="machine-1-1", help="SMD machine id"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = ExperimentConfig(machine_id=args.machine_id)
    cfg.experiment_name = f"smd_{cfg.machine_id}"
    device = torch.device(cfg.device)

    exp_dir = cfg.experiment_dir()
    ckpt_path = exp_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load data
    train_raw, test_raw, test_labels = load_smd_machine(
        root_dir=cfg.dataset_root, machine_id=cfg.machine_id
    )
    if cfg.normalize:
        mean, std = compute_zscore_params(train_raw)
        test_data = apply_zscore(test_raw, mean, std)
    else:
        test_data = test_raw

    test_ds = SlidingWindowDataset(
        series=test_data,
        labels=test_labels,
        window_size=cfg.window_size,
        stride=cfg.window_stride,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Rebuild encoder
    encoder = TimeSeriesTransformerAE(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            x_recon, _ = encoder(x)
            mse = (x_recon - x).pow(2).mean(dim=(1, 2))  # (B,)

            scores.append(mse.cpu().numpy())
            window_labels = (y.max(dim=1).values > 0).long().cpu().numpy()
            labels.append(window_labels)

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    metrics = evaluate_window_metrics(scores, labels)
    print("Evaluation metrics (window-level):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
