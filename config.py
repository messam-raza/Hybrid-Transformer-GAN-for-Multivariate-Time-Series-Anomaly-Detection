import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SMD_DIR = os.path.join(DATA_DIR, "ServerMachineDataset")


@dataclass
class SMDConfig:
    # Root directory of SMD dataset
    root_dir: str = SMD_DIR
    # Sliding window parameters
    window_size: int = 100
    step_size: int = 10
    # Whether to apply z-score normalization
    normalize: bool = True
    # If empty, use all machines. Otherwise, list machine names (file stems without .txt)
    machines: tuple[str, ...] = ()


@dataclass
class ModelConfig:
    # SMD has 38 metrics per time step
    input_dim: int = 38
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    latent_dim: int = 64  # for contrastive + GAN


@dataclass
class TrainConfig:
    batch_size: int = 64         # shrink to 32 if memory is low
    num_epochs: int = 20
    lr: float = 1e-3
    device: str = "cpu"          # will be set dynamically
    recon_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.5
    gan_loss_weight: float = 0.5
