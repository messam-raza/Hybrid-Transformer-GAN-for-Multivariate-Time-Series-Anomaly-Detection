import torch
import torch.nn as nn
import torch.nn.functional as F


mse_loss_fn = nn.MSELoss()


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between reconstruction and target.
    """
    return mse_loss_fn(recon, target)


def gan_discriminator_loss(
    d_real_logits: torch.Tensor,
    d_fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Non-saturating GAN discriminator loss with BCE logits.
    """
    real_labels = torch.ones_like(d_real_logits)
    fake_labels = torch.zeros_like(d_fake_logits)

    loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)

    return loss_real + loss_fake


def gan_generator_loss(
    d_fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Generator loss: fool the discriminator (labels = 1).
    """
    real_labels = torch.ones_like(d_fake_logits)
    return F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss for contrastive learning between two views.

    z1, z2 : (B, D)
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # (2B, D)

    similarity_matrix = representations @ representations.t()  # (2B, 2B)
    similarity_matrix = similarity_matrix / temperature

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    # Positive index for each row:
    # for i in [0..B-1], positive is i+B
    # for i in [B..2B-1], positive is i-B
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
