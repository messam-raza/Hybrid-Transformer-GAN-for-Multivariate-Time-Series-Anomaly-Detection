import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.config import SMDConfig, ModelConfig, TrainConfig
from src.data.preprocessing import compute_smd_normalization_stats
from src.data.smd_dataset import SMDSequenceDataset
from src.models.hybrid_model import HybridAnomalyModel
from src.utils.metrics import compute_roc_auc
from src.utils.plot import save_reconstruction_plot
from src.utils.seed import set_seed


def load_dataloader(
    smd_cfg: SMDConfig,
    batch_size: int,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    mean, std = compute_smd_normalization_stats(smd_cfg.root_dir, smd_cfg.machines)
    test_dataset = SMDSequenceDataset(smd_cfg, split="test", mean=mean, std=std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    return test_loader, mean_t, std_t


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
) -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    smd_cfg = SMDConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    train_cfg.device = str(device)

    test_loader, mean_t, std_t = load_dataloader(smd_cfg, train_cfg.batch_size)

    model = HybridAnomalyModel(
        input_dim=model_cfg.input_dim,
        window_size=smd_cfg.window_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        latent_dim=model_cfg.latent_dim,
    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores = []
    all_labels = []

    for i, batch in enumerate(test_loader):
        x = batch["input"].to(device)
        y = batch["label"].to(device)

        recon, z = model.transformer(x)
        recon_error = torch.mean((recon - x) ** 2, dim=(1, 2))

        all_scores.append(recon_error.cpu())
        all_labels.append(y.cpu())

        if i == 0:
            save_reconstruction_plot(
                x.cpu(),
                recon.cpu(),
                save_path=os.path.join("outputs", "figures", "reconstruction_example.png"),
            )

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    auc = compute_roc_auc(labels, scores)
    print(f"Checkpoint ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    ckpt_path = os.path.join("outputs", "checkpoints", "best_model.pt")
    evaluate_checkpoint(ckpt_path)
