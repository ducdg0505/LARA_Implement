LARA-TSAD: Light and Anti-overfitting Retraining Approach for Time-Series Anomaly Detection

This repository contains the official implementation of LARA, a robust Test-Time Adaptation (TTA) framework designed for Time-Series Anomaly Detection (TSAD). It directly addresses the critical challenge of Distribution Shift (e.g., sensor drift, environmental changes) while preventing Model Poisoning.

Version 25.0 introduces three novel algorithmic enhancements—Clean Memory Initialization (CMI), MAD-FES Profiling, and Micro-Retraining—achieving state-of-the-art performance with an Average F1-Score of 0.9324 on the Server Machine Dataset (SMD).

Part 1: Architectural Framework & Codebase Structure

The system is designed with a strict modular architecture to separate data engineering, representation learning, test-time adaptation, and evaluation mechanisms.

config.yaml: The centralized configuration hub. It defines all architectural hyperparameters, optimization variables, and evaluation metrics (e.g., sliding window size, latent dimensions, retraining ratios) to ensure experimental reproducibility.

data_utils.py: Manages the data ingestion pipeline. It handles deterministic Min-Max normalization and constructs sliding-window tensors using memory-efficient memory strides (torch.as_strided), mitigating RAM bottlenecks during high-frequency time-series processing.

base_model.py: Contains the foundational Base VAE (Variational Autoencoder). Utilizing a GRU/LSTM-based Encoder-Decoder topology, this module is strictly pre-trained on historical data to map normal temporal dynamics into a Gaussian posterior distribution.

ruminate.py: Implements the RuminateBlock. This module operates as an associative memory mechanism. It computes a target latent manifold ($\tilde{Z}$) for novel test observations via Monte Carlo estimation, weighted by the likelihood of historically retrieved samples.

lara_wrapper.py: The core adaptation wrapper. It freezes the Base VAE parameters and injects two lightweight, linear affine transformation matrices: $M_z$ (Latent Space Adjustment) and $M_x$ (Output Space Adjustment).

trainer.py: Encapsulates the optimization routines. It includes the BaseTrainer (optimizing the Evidence Lower Bound - ELBO) and the LARARetrainer (solving the convex adjustment objectives for $M_z$ and $M_x$).

detector.py: The evaluation and scoring engine. It computes discrepancy metrics, executes Extreme Value Theory (EVT) based thresholding via the Peak-Over-Threshold (POT) algorithm, and applies Point-Adjustment (PA) protocols standard in TSAD literature.

main.py: The primary execution pipeline. It orchestrates the end-to-end experimental workflow: Data Loading $\rightarrow$ Base Prior Learning $\rightarrow$ LARA Test-Time Adaptation $\rightarrow$ Inference $\rightarrow$ Performance Profiling.

Part 2: Methodological Innovations (V25.0 Breakthroughs)

The significant leap in performance (F1 > 0.93) in V25.0 is attributed to three mathematically grounded mechanisms designed to fortify the model against outliers and representation degradation.

2.1 Clean Memory Initialization (CMI)

In conventional setups, historical buffer retrieval heavily relies on random sampling from the training corpus, which inevitably propagates latent noise or edge-case anomalies into the target estimation phase.

Mechanism: CMI employs the pre-trained Base VAE to evaluate the entire historical buffer through a rigorous reconstruction scoring phase. It applies a Quantile-based Pruning strategy, systematically discarding the 10% of samples with the highest reconstruction errors.

Impact: Only the top 90% most pristine, normative samples are committed to the RuminateBlock's memory. This guarantees that when LARA computes semantic distances for target estimation, it references a "Purified Prior", yielding highly stable and noise-free gradient signals during fine-tuning.

2.2 Robust MAD-FES (Median Absolute Deviation - Final Error Scoring)

Traditional TSAD methodologies often suffer from anomaly masking due to standard Z-score normalizations and window-averaged MSE formulations. MAD-FES resolves this through a dual-pronged scoring paradigm:

Final Error Scoring (FES) - Mitigating Error Dilution: Averaging errors across a temporal window dilutes localized anomalies, burying them beneath historical normal points. FES extracts reconstruction and latent discrepancies strictly at the last time-step of the sequence ($t_W$). This ensures instantaneous fault localization without temporal smoothing.

MAD Normalization - Resisting Outlier Distortion: Standard parametric moments (Mean $\mu$, Standard Deviation $\sigma$) are notoriously sensitive to extreme outliers, leading to threshold inflation. MAD-FES substitutes these with robust rank statistics: the Median and the Median Absolute Deviation (MAD).

Impact: The resulting anomaly threshold is highly resilient (robust) to extreme environmental noise, drastically minimizing False Negatives and elevating the overall Recall without sacrificing Precision.

2.3 Micro-Retraining (1% Adaptation Buffer)

Conventional Test-Time Adaptation assumes "more data yields better adaptation." However, in unsupervised anomaly detection, incorporating large segments of test data exponentially increases the probability of ingesting anomalies into the retraining loop.

Mechanism: The adaptation buffer ratio (retrain_ratio) is drastically restricted to merely 1% ($\tau = 0.01$) of the initial test stream.

Impact (The Double-Shield Effect): 1. Anti-Poisoning: A hyper-restricted buffer severely limits the integration of anomalous instances during the fine-tuning phase, preventing the model from normalizing abnormal behaviors (Model Poisoning).
2. Anti-Catastrophic Forgetting: Because the LARA modules ($M_z, M_x$) are inherently linear transformations, they converge optimally with minimal sample sizes. The 1% buffer acts as a natural regularizer, providing just enough signal to align the distribution axes to the new environment without overwriting the deep, foundational representations learned by the Base VAE.
