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

```text
LARA-TSAD/
├── data/                   # Directory for datasets (e.g., SMD, MSL, SMAP)
├── base_model.py           # Core VAE architecture (Encoder-Decoder)
├── config.yaml             # Centralized hyperparameter configuration
├── data_utils.py           # Memory-efficient sliding window data loaders
├── detector.py             # EVT/POT thresholding and evaluation metrics
├── lara_wrapper.py         # The LARA adaptation modules (Mz, Mx)
├── main.py                 # Main experimental pipeline orchestration
├── ruminate.py             # Associative memory module via Monte Carlo estimation
└── trainer.py              # Base ELBO optimization and Convex adaptation logic
