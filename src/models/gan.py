import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesGenerator(nn.Module):
    """
    Lightweight MLP-based generator that maps latent vectors to flattened windows.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, latent_dim) -> (B, output_dim)
        """
        return self.net(z)


class TimeSeriesDiscriminator(nn.Module):
    """
    Lightweight MLP-based discriminator for flattened windows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat : (B, input_dim) -> logits (B, 1)
        """
        return self.net(x_flat)
