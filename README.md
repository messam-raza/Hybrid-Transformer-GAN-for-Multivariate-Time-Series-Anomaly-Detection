# Hybrid Transformer–GAN for Multivariate Time-Series Anomaly Detection

This repository contains the official implementation of a **Hybrid Transformer–GAN framework** for robust **multivariate time-series anomaly detection**, designed specifically to handle **contaminated training data**, improve **generalization**, and enhance **representation learning** through:

- **Geometric masking**  
- **Transformer-based temporal feature extraction**  
- **Contrastive representation learning**  
- **Generative Adversarial Networks (GANs)**

This work aligns with modern research directions in AI, anomaly detection, and AIOps, and can be extended into a **conference paper**.

---

# 🔍 Overview

Traditional anomaly detection models struggle when the training data contains hidden anomalies.  
This framework integrates **reconstruction**, **contrastive learning**, and **generative modeling** to produce:

- More robust latent representations  
- Better reconstruction error signals  
- Improved anomaly separation  
- Higher stability on noisy/mixed datasets  

---

# 📁 Project Structure
Hybrid-Transformer-GAN-for-Multivariate-Time-Series-Anomaly-Detection/
│
├── config.yaml
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│ └── ServerMachineDataset/
│ ├── train/
│ ├── test/
│ ├── test_label/
│ └── interpretation_label/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_model_debug.ipynb
│ └── 03_final_training.ipynb
│
├── outputs/
│ ├── checkpoints/
│ ├── figures/
│ └── logs/
│
└── src/
├── config.py
│
├── data/
│ ├── preprocessing.py
│ ├── smd_dataset.py
│ ├── masking.py
│ └── init.py
│
├── models/
│ ├── transformer.py
│ ├── gan.py
│ ├── hybrid_model.py
│ └── init.py
│
├── training/
│ ├── train.py
│ ├── eval.py
│ ├── losses.py
│ └── init.py
│
└── utils/
├── seed.py
├── plot.py
├── metrics.py
└── init.py


---

# 📦 Installation

### Clone the repository

```bash
git clone <your-private-repo-url>
cd Hybrid-Transformer-GAN-for-Multivariate-Time-Series-Anomaly-Detection

Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows


Install dependencies

pip install -r requirements.txt

# OR install as a package
pip install -e .

📊 Dataset

This project uses the Server Machine Dataset (SMD) from the KDD 2019 paper
“Robust Anomaly Detection for Multivariate Time Series.”

Expected structure:
data/ServerMachineDataset/
│── train/
│── test/
│── test_label/
│── interpretation_label/

Each file is a .txt representing a multivariate sensor stream.

🚀 Training

Run the full training loop:

python -m src.training.train


This will:

Train Transformer encoder

Train GAN (Generator + Discriminator)

Apply masking + contrastive objectives

Save the best checkpoint to:

outputs/checkpoints/best_model.pt


📈 Evaluation

Evaluate the best model:

python -m src.training.eval

This evaluates:

Reconstruction error

ROC-AUC score

Saves reconstruction plots in:

outputs/figures/


Methodology Summary
✔ Transformer Encoder

Extracts deep temporal patterns and produces latent embeddings.

✔ Contrastive Loss (InfoNCE)

Forces masked views of the same window to produce similar representations → improves robustness.

✔ GAN

The Generator learns realistic windows;
The Discriminator stabilizes latent space quality.

✔ Reconstruction Loss

Used for anomaly scoring:
High MSE → higher probability of anomaly.


📉 Metrics

The framework outputs:

Reconstruction Error

ROC-AUC

Precision / Recall / F1 (optional extension)

Reconstruction Plots


🖼 Example Outputs

Saved under:

outputs/figures/reconstruction.png

Shows original vs reconstructed features for qualitative analysis.


🛠 Extendability

This project is structured for easy research extensions:

Replace GAN with VAE

Add temporal convolution modules

Add anomaly heatmaps

Add adaptive thresholding (SPOT/Peak-over-threshold)

Support other datasets: SMAP, MSL, SKAB