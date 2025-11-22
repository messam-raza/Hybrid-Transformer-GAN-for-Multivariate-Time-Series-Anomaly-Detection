from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SMD_DIR = DATA_DIR / "ServerMachineDataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class ExperimentConfig:
    # ---- Dataset ----
    dataset_root: Path = SMD_DIR
    machine_id: str = "machine-1-1"
    window_size: int = 100
    window_stride: int = 1
    normalize: bool = True
    train_val_split: float = 0.9

    # ---- Model ----
    input_dim: int = 38           # SMD has 38 metrics
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    latent_dim: int = 128

    # ---- Training ----
    batch_size: int = 64
    num_epochs: int = 20
    lr_main: float = 1e-4
    lr_discriminator: float = 5e-5
    weight_decay: float = 1e-5
    num_workers: int = 2

    # ---- Loss weights / temperature ----
    lambda_recon: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_gan: float = 0.1
    contrastive_temperature: float = 0.2

    # ---- Device & naming ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name: str = "smd_machine-1-1"

    def experiment_dir(self) -> Path:
        path = OUTPUT_DIR / "checkpoints" / self.experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figures_dir(self) -> Path:
        path = OUTPUT_DIR / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def logs_dir(self) -> Path:
        path = OUTPUT_DIR / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
