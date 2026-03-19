# LARA-Enhanced: Robust Time-Series Anomaly Detection (F1 > 0.93)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an advanced, highly optimized implementation of the **LARA** (Light and Anti-overfitting Retraining Approach) model for Time-Series Anomaly Detection. 

Through a series of structural enhancements—namely **Clean Memory Initialization (CMI)**, **MAD-FES (Median Absolute Deviation Feature Error Scaling)**, and **Multi-Scale Ensemble**—this implementation achieves a state-of-the-art **Average F1-Score of 0.9324** (Median F1: 0.9740) on the Server Machine Dataset (SMD).

## 🚀 Key Innovations & Enhancements (V25.0)

While the base LARA paper provides a strong theoretical foundation with its linear $M_x$ and $M_z$ adjustment layers, real-world datasets like SMD contain hidden anomalies in the training set and exhibit massive variance across different machines. We solved this with three core techniques:

### 1. Clean Memory Initialization (CMI)
In the standard Ruminate Block, historical samples are randomly selected. However, if the training set contains anomalies, the target latent vector $\tilde{z}$ becomes corrupted.
* **How it works:** Before initializing the Ruminate Block's memory, we sample 2,000 candidates from the historical data. We run these through the frozen Base VAE and calculate their reconstruction errors. We then **filter out the top 10% with the highest errors**, ensuring the history memory consists *only* of highly pure, normal samples.

### 2. Robust MAD-FES (Median Absolute Deviation - Feature Error Scaling)
Standard normalization (Mean/Std or Min/Max) is heavily skewed by extreme anomaly values during evaluation. 
* **How it works:** We replaced standard scaling with MAD (Median Absolute Deviation). 
* **Smart Flooring:** To prevent division by zero for highly stable features, we calculate a dynamic `floor = median(all_MADs) * 0.05`. This aggressively isolates true anomalies without exploding the gradients of minor fluctuations.
* **Dual-Space Scoring:** The final anomaly score is a weighted combination of both Reconstruction Error and Latent Space Shift (`total_err = recon_err + 0.5 * latent_err`).

### 3. Multi-Scale Context Ensemble
Single window sizes often miss the broader context or over-smooth sharp spikes.
* **How it works:** The model trains parallel pipelines on different window scales (e.g., `Window Size = 32` and `Window Size = 96`). The anomaly scores are log-scaled (`log1p`), normalized, and then averaged. This captures both short-term glitches and long-term contextual drift.

---

## 📂 Repository Structure

The codebase is highly modularized for research and production deployment:

```text
.
├── main.py               # Main pipeline: orchestrates loading, training, and evaluation
├── config.yaml           # Centralized configuration (hyperparameters, LARA configs)
├── data_utils.py         # DatasetManager: Sliding window generation and MinMaxScaler
├── base_model.py         # Base VAE architecture (GRU-based Encoder/Decoder)
├── lara_wrapper.py       # LARA wrapper: Implements Theorem 1 (Linear Mx, Mz layers)
├── ruminate.py           # Ruminate Block: Monte Carlo sampling and CMI integration
├── trainer.py            # BaseTrainer (ELBO loss) & LARARetrainer (Convex loss)
└── detector.py           # AnomalyDetector: Point-adjustment heuristics and F1 evaluation
