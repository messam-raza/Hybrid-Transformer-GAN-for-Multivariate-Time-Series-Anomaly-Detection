from typing import Tuple

import torch
import torch.nn as nn

from src.models.transformer import TimeSeriesTransformer
from src.models.gan import TimeSeriesGenerator, TimeSeriesDiscriminator


class HybridAnomalyModel(nn.Module):
    """
    Hybrid model combining:
    - Transformer-based reconstruction and representation
    - GAN-based generation and discrimination
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.transformer = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim,
        )

        flat_dim = input_dim * window_size
        self.generator = TimeSeriesGenerator(
            latent_dim=latent_dim,
            output_dim=flat_dim,
        )
        self.discriminator = TimeSeriesDiscriminator(
            input_dim=flat_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x       : (B, T, D)
        z_noise : (B, latent_dim)

        Returns:
        recon          : (B, T, D)
        z              : (B, latent_dim)
        d_real_logits  : (B, 1)
        d_fake_logits  : (B, 1)
        """
        recon, z = self.transformer(x)

        B, T, D = x.shape
        assert T == self.window_size and D == self.input_dim, \
            "Input shape mismatch with configured window size / input dim."

        x_flat = x.view(B, -1)

        # Generator uses latent representation plus noise
        z_g = z + z_noise
        x_fake_flat = self.generator(z_g)

        d_real_logits = self.discriminator(x_flat)
        d_fake_logits = self.discriminator(x_fake_flat.detach())

        return recon, z, d_real_logits, d_fake_logits

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic sequences from latent vectors.
        z : (B, latent_dim) -> (B, T, D)
        """
        x_flat = self.generator(z)
        B = x_flat.size(0)
        x = x_flat.view(B, self.window_size, self.input_dim)
        return x
