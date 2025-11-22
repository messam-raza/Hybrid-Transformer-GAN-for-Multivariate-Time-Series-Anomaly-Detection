from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TimeSeriesTransformerAE(nn.Module):
    """
    Transformer-based autoencoder.

    Input:  (B, T, D_in)
    Output: reconstruction (B, T, D_in), latent (B, d_model)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, input_dim)
        self.latent_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D_in)
        Returns:
            x_recon: (B, T, D_in)
            z: (B, d_model)
        """
        h = self.input_projection(x)   # (B, T, d_model)
        h = self.positional_encoding(h)
        h = self.encoder(h)            # (B, T, d_model)

        # Mean pool
        z = h.mean(dim=1)
        z = self.latent_norm(z)

        x_recon = self.output_projection(h)
        return x_recon, z


# Alias to avoid import confusion
TimeSeriesTransformer = TimeSeriesTransformerAE
