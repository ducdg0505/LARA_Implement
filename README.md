# LARA-TSAD: Light and Anti-overfitting Retraining Approach for Time-Series Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-25.0-success.svg)]()

This repository contains the official PyTorch implementation of **LARA-TSAD**, a robust Test-Time Adaptation (TTA) framework designed for unsupervised Time-Series Anomaly Detection under severe distribution shifts.

## Abstract

Time-series anomaly detection (TSAD) in real-world cyber-physical systems is often severely compromised by non-stationary environments, such as sensor drift or operational state transitions. Traditional static models suffer from catastrophic performance degradation when confronted with these distribution shifts. We propose **LARA** (Light and Anti-overfitting Retraining Approach), a mathematically principled framework that wraps around pre-trained generative models (e.g., Variational Autoencoders) to enable real-time adaptation without catastrophic forgetting. 

**Version 25.0** introduces three breakthrough mechanisms: **Clean Memory Initialization (CMI)**, **MAD-FES Profiling**, and **Micro-Retraining**, pushing the state-of-the-art boundaries on the Server Machine Dataset (SMD) to an Average F1-Score of **0.9324**.

---

## Repository Structure

* **`config.yaml`**: Centralized hyperparameter configuration.
* **`data_utils.py`**: Memory-efficient sliding window data loaders via `torch.as_strided`.
* **`base_model.py`**: Core VAE architecture (Encoder-Decoder) for prior representation learning.
* **`ruminate.py`**: Associative memory module via Monte Carlo estimation.
* **`lara_wrapper.py`**: The LARA adaptation layers ($M_z$, $M_x$).
* **`trainer.py`**: Base ELBO optimization and Convex adaptation logic.
* **`detector.py`**: EVT/POT thresholding and Point-Adjustment evaluation metrics.
* **`main.py`**: Main experimental pipeline orchestration.

---

## Highlights of Version 25.0

* **Robustness against Model Poisoning:** Utilizes a restricted 1% adaptation buffer to prevent anomalies from corrupting the adaptive process.
* **Convex Optimization:** Adapts exclusively via low-dimensional affine transformations ($M_z$, $M_x$), guaranteeing fast and stable convergence during test-time.
* **Instantaneous Fault Localization:** Replaces window-averaged error formulations with Final Error Scoring (FES) to preserve the temporal resolution of sudden anomalies.

---

## Getting Started

This guide provides step-by-step instructions to set up the environment, download necessary datasets, and reproduce the benchmark results.

### 1. Environment Setup

The codebase is built on Python 3.8+ and PyTorch. We recommend using a virtual environment (e.g., Anaconda or Python `venv`).

```bash
# Clone the repository
git clone [https://github.com/your-username/LARA-TSAD.git](https://github.com/your-username/LARA-TSAD.git)
cd LARA-TSAD

# Create virtual environment
python -m venv lara_env
source lara_env/bin/activate  # On Windows use: lara_env\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn scipy pyyaml tqdm
