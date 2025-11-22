from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data.smd_dataset import load_smd_machine, SlidingWindowDataset
from src.data.preprocessing import compute_zscore_params, apply_zscore
from src.data.masking import geometric_mask
from src.models.transformer import TimeSeriesTransformerAE
from src.models.gan import LatentGenerator, LatentDiscriminator
from src.training.losses import reconstruction_loss, info_nce_loss, gan_losses
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Transformer-GAN TSAD on SMD."
    )
    parser.add_argument(
        "--machine_id", type=str, default="machine-1-1", help="SMD machine id"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = ExperimentConfig(machine_id=args.machine_id)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.experiment_name = f"smd_{cfg.machine_id}"

    device = torch.device(cfg.device)
    exp_dir = cfg.experiment_dir()
    print(f"Using device: {device}")
    print(f"Experiment dir: {exp_dir}")

    # 1. Load raw data
    train_raw, test_raw, test_labels = load_smd_machine(
        root_dir=cfg.dataset_root, machine_id=cfg.machine_id
    )
    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

    # 2. Normalize
    if cfg.normalize:
        mean, std = compute_zscore_params(train_raw)
        train_data = apply_zscore(train_raw, mean, std)
        test_data = apply_zscore(test_raw, mean, std)
    else:
        train_data = train_raw
        test_data = test_raw

    # 3. Datasets
    full_train_ds = SlidingWindowDataset(
        series=train_data,
        labels=None,
        window_size=cfg.window_size,
        stride=cfg.window_stride,
    )
    n_train = int(len(full_train_ds) * cfg.train_val_split)
    n_val = len(full_train_ds) - n_train
    train_ds, val_ds = random_split(
        full_train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    test_ds = SlidingWindowDataset(
        series=test_data,
        labels=test_labels,
        window_size=cfg.window_size,
        stride=cfg.window_stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # 4. Models
    encoder = TimeSeriesTransformerAE(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)

    generator = LatentGenerator(cfg.latent_dim).to(device)
    discriminator = LatentDiscriminator(cfg.latent_dim).to(device)

    # 5. Optimizers
    opt_main = torch.optim.AdamW(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=cfg.lr_main,
        weight_decay=cfg.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.lr_discriminator,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = None

    # 6. Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = run_epoch(
            encoder,
            generator,
            discriminator,
            train_loader,
            opt_main,
            opt_d,
            cfg,
            device,
            epoch,
            is_train=True,
        )
        val_loss = run_epoch(
            encoder,
            generator,
            discriminator,
            val_loader,
            opt_main,
            opt_d,
            cfg,
            device,
            epoch,
            is_train=False,
        )

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f}"
        )

        # Save best
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = exp_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": cfg.__dict__,
                    "encoder": encoder.state_dict(),
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    # Sanity: one test batch
    print("Computing reconstruction errors on test set (sanity check)...")
    encoder.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon, _ = encoder(x)
            _ = (x_recon - x).pow(2).mean().item()
            break

    print("Training finished.")


def run_epoch(
    encoder,
    generator,
    discriminator,
    dataloader,
    opt_main,
    opt_d,
    cfg: ExperimentConfig,
    device,
    epoch: int,
    is_train: bool,
):
    if is_train:
        encoder.train()
        generator.train()
        discriminator.train()
    else:
        encoder.eval()
        generator.eval()
        discriminator.eval()

    total_loss = 0.0
    count = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [{'train' if is_train else 'val'}]")

    for x, _ in loop:
        x = x.to(device)  # (B, T, D)
        batch_size = x.size(0)

        x_aug = geometric_mask(x)

        # Forward encoder
        x_recon, z_orig = encoder(x)
        x_recon_aug, z_aug = encoder(x_aug)

        if is_train:
            # --- Update discriminator ---
            opt_d.zero_grad(set_to_none=True)

            z_fake_for_d = generator(z_aug)
            d_loss, _ = gan_losses(discriminator, z_real=z_orig, z_fake=z_fake_for_d)
            d_loss.backward()
            opt_d.step()

            # --- Update encoder + generator ---
            opt_main.zero_grad(set_to_none=True)
            z_fake = generator(z_aug)

            recon_loss_orig = reconstruction_loss(x, x_recon)
            recon_loss_aug = reconstruction_loss(x, x_recon_aug)
            recon_total = 0.5 * (recon_loss_orig + recon_loss_aug)

            cont_loss = info_nce_loss(
                z_orig, z_aug, temperature=cfg.contrastive_temperature
            )

            _, g_loss = gan_losses(discriminator, z_real=z_orig, z_fake=z_fake)

            loss = (
                cfg.lambda_recon * recon_total
                + cfg.lambda_contrastive * cont_loss
                + cfg.lambda_gan * g_loss
            )

            loss.backward()
            opt_main.step()

        else:
            with torch.no_grad():
                z_fake = generator(z_aug)
                recon_loss_orig = reconstruction_loss(x, x_recon)
                recon_loss_aug = reconstruction_loss(x, x_recon_aug)
                recon_total = 0.5 * (recon_loss_orig + recon_loss_aug)
                cont_loss = info_nce_loss(
                    z_orig, z_aug, temperature=cfg.contrastive_temperature
                )
                _, g_loss = gan_losses(discriminator, z_real=z_orig, z_fake=z_fake)
                loss = (
                    cfg.lambda_recon * recon_total
                    + cfg.lambda_contrastive * cont_loss
                    + cfg.lambda_gan * g_loss
                )

        total_loss += loss.item()
        count += 1

        loop.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_total.item():.4f}",
                "B": batch_size,
            }
        )

    return total_loss / max(count, 1)


if __name__ == "__main__":
    main()
