import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.config import SMDConfig, ModelConfig, TrainConfig
from src.data.preprocessing import compute_smd_normalization_stats
from src.data.smd_dataset import SMDSequenceDataset
from src.data.masking import apply_geometric_mask
from src.models.hybrid_model import HybridAnomalyModel
from src.training.losses import (
    reconstruction_loss,
    gan_discriminator_loss,
    gan_generator_loss,
    info_nce_loss,
)
from src.utils.seed import set_seed
from src.utils.metrics import compute_roc_auc
from src.utils.plot import save_reconstruction_plot


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_dataloaders(
    smd_config: SMDConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    mean, std = compute_smd_normalization_stats(smd_config.root_dir, smd_config.machines)

    train_dataset = SMDSequenceDataset(smd_config, split="train", mean=mean, std=std)
    test_dataset = SMDSequenceDataset(smd_config, split="test", mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)

    return train_loader, test_loader, mean_t, std_t


def train_one_epoch(
    model: HybridAnomalyModel,
    train_loader: DataLoader,
    optimizer_g: Adam,
    optimizer_d: Adam,
    device: torch.device,
    train_cfg: TrainConfig,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in train_loader:
        x = batch["input"].to(device)  # (B, T, D)
        B, T, D = x.shape

        z_noise = torch.randn(B, model.latent_dim, device=device)

        # ----- Discriminator update -----
        optimizer_d.zero_grad()
        with torch.no_grad():
            _, _, d_real_logits, d_fake_logits = model(x, z_noise)
        d_loss = gan_discriminator_loss(d_real_logits, d_fake_logits)
        d_loss.backward()
        optimizer_d.step()

        # ----- Generator + Transformer + Contrastive update -----
        optimizer_g.zero_grad()

        recon, z, d_real_logits, d_fake_logits = model(x, z_noise)

        # Reconstruction loss
        recon_loss = reconstruction_loss(recon, x)

        # Contrastive loss using two masked views
        x_masked1, _ = apply_geometric_mask(x, mask_type="random_rectangle", mask_ratio=0.15)
        x_masked2, _ = apply_geometric_mask(x, mask_type="random_rectangle", mask_ratio=0.15)

        _, z1 = model.transformer(x_masked1)
        _, z2 = model.transformer(x_masked2)

        contrastive = info_nce_loss(z1, z2, temperature=0.2)

        # Generator loss (GAN)
        g_loss = gan_generator_loss(d_fake_logits)

        loss = (
            train_cfg.recon_loss_weight * recon_loss
            + train_cfg.contrastive_loss_weight * contrastive
            + train_cfg.gan_loss_weight * g_loss
        )

        loss.backward()
        optimizer_g.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: HybridAnomalyModel,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_scores = []
    all_labels = []

    for i, batch in enumerate(test_loader):
        x = batch["input"].to(device)
        y = batch["label"].to(device)

        recon, z = model.transformer(x)
        recon_error = torch.mean((recon - x) ** 2, dim=(1, 2))  # per-window score

        all_scores.append(recon_error.cpu())
        all_labels.append(y.cpu())

        if i == 0:
            save_reconstruction_plot(x.cpu(), recon.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    auc = compute_roc_auc(labels, scores)
    print(f"Test ROC-AUC: {auc:.4f}")
    return auc


def main() -> None:
    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    smd_cfg = SMDConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    train_cfg.device = str(device)

    train_loader, test_loader, mean_t, std_t = create_dataloaders(smd_cfg, train_cfg.batch_size)

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

    optimizer_g = Adam(
        list(model.transformer.parameters()) + list(model.generator.parameters()),
        lr=train_cfg.lr,
    )
    optimizer_d = Adam(model.discriminator.parameters(), lr=train_cfg.lr)

    best_auc = 0.0
    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(1, train_cfg.num_epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer_g,
            optimizer_d,
            device,
            train_cfg,
            epoch,
        )
        auc = evaluate(model, test_loader, device)

        if auc > best_auc:
            best_auc = auc
            ckpt_path = os.path.join("outputs", "checkpoints", "best_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "auc": best_auc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path}")

    print(f"Training complete. Best ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
