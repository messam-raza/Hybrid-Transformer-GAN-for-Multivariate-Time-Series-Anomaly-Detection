from setuptools import setup, find_packages

setup(
    name="hybrid_transformer_gan_tsad",
    version="1.0.0",
    author="Messam Raza",
    description="Hybrid Transformer–GAN Framework for Multivariate Time-Series Anomaly Detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.9",
)
