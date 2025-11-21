from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        return x + self.pe[:, :T, :D]


class TimeSeriesTransformer(nn.Module):
    """
    Lightweight Transformer for multivariate time-series
    reconstruction and representation.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.latent_proj = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, D)

        Returns:
        recon : (B, T, D)
        z     : (B, latent_dim)  - mean-pooled latent
        """
        h = self.input_proj(x)        # (B, T, d_model)
        h = self.pos_encoding(h)
        h = self.encoder(h)           # (B, T, d_model)

        z = h.mean(dim=1)             # (B, d_model)
        z = self.latent_proj(z)       # (B, latent_dim)

        B, T, _ = x.shape
        z_expanded = z.unsqueeze(1).expand(B, T, self.latent_dim)
        recon = self.decoder(z_expanded)  # (B, T, D)

        return recon, z
