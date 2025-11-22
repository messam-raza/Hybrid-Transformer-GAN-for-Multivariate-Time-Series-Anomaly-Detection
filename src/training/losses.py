from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


mse = nn.MSELoss(reduction="mean")
bce_logits = nn.BCEWithLogitsLoss(reduction="mean")


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return mse(x_recon, x)


def info_nce_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    InfoNCE loss for a batch of positive pairs (z_i, z_j).
    """
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    logits = (z_i @ z_j.t()) / temperature
    labels = torch.arange(z_i.size(0), device=z_i.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_j)


def gan_losses(
    discriminator: nn.Module,
    z_real: torch.Tensor,
    z_fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (d_loss, g_loss)
    """
    logits_real = discriminator(z_real.detach())
    logits_fake = discriminator(z_fake.detach())

    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)

    d_loss_real = bce_logits(logits_real, real_labels)
    d_loss_fake = bce_logits(logits_fake, fake_labels)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    logits_fake_for_g = discriminator(z_fake)
    g_loss = bce_logits(logits_fake_for_g, real_labels)

    return d_loss, g_loss
