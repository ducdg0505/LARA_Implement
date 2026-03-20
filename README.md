# Project Report: Robust Time-Series Anomaly Detection using Enhanced LARA (V25.0)

**Project Objective:** To achieve highly accurate anomaly detection on the Server Machine Dataset (SMD) by overcoming training noise and evaluation metric sensitivities, targeting an Average F1-Score above 0.90.

## 1. Executive Summary
This project implements and significantly enhances the **LARA** (Light and Anti-overfitting Retraining Approach) model for Time-Series Anomaly Detection. While the base LARA model utilizes a Variational Autoencoder (VAE) and linear adjustments to adapt to new data distributions, it struggles with noisy training data and extreme anomaly spikes during evaluation. 

By introducing **Version 25 (V25.0)**, this project integrates three major architectural and mathematical improvements, successfully pushing the Average F1-Score to **0.9324** on the complete SMD benchmark.

## 2. Key Architectural Innovations (Version 25)
The `Version25.py` implementation achieves state-of-the-art results through the following techniques:

* **Clean Memory Initialization (CMI):** In standard LARA, historical memory for the Ruminate Block is sampled randomly. In V25, we evaluate 2,000 historical candidates using the frozen Base VAE and actively filter out the top 10% with the highest reconstruction errors. This guarantees the target latent vector is guided only by strictly "normal" data.
* **Robust MAD-FES (Median Absolute Deviation - Feature Error Scaling):** Standard scaling methods (Min-Max or Mean/Std) are easily distorted by massive anomaly spikes. V25 replaces these with MAD. Furthermore, we implemented a "Smart Flooring" technique (setting a minimum bound based on the median of all MADs) to prevent gradient explosions on highly stable features.
* **Multi-Scale Context Ensemble:** Instead of relying on a single window size, the model evaluates time-series data using parallel pipelines at different window scales (e.g., 32 and 96). The normalized, log-scaled anomaly scores are then averaged, capturing both sudden micro-glitches and prolonged contextual drifts.

## 3. Project Structure
The repository is modularly designed for research and deployment. Below are the core components:

* **`Version25.py`:** The primary execution script containing the fully enhanced LARA model (CMI, MAD-FES, Ensemble). This is the file that achieves the 0.93+ F1 score.
* **`base_model.py`:** Contains the architecture for the Base VAE (GRU-based Encoder and Decoder).
* **`config.yaml`:** The central configuration file for all hyperparameters, model dimensions, and LARA specific settings.
* **`data_utils.py`:** Handles data loading, MinMaxScaler normalization, and sliding window generation.
* **`lara_wrapper.py`:** Implements the LARA logic (freezing the base VAE and updating the linear adjustment layers).
* **`ruminate.py`:** Contains the logic for Monte Carlo sampling to estimate the target latent representations.

## 4. Hyperparameter Configuration (`config.yaml`)
Before running the models, you can adjust the system settings and hyperparameters in the `config.yaml` file. 

## 5. Step-by-Step Execution Guide

### Step 1: Environment Setup
Ensure you have Python 3.8 or higher installed. Install the required libraries via terminal or command prompt:

```bash
pip install torch numpy pandas scikit-learn pyyaml tqdm
```

### Step 2: Data Preparation
Ensure the Server Machine Dataset (SMD) is correctly placed in the project directory as defined in the `config.yaml` file (default is `data/SMD/`). The required folder structure is as follows:
* `data/SMD/train/`: Contains the `.txt` training data files (normal data).
* `data/SMD/test/`: Contains the testing data files (containing anomalies).
* `data/SMD/test_label/`: Contains the ground truth label files (0/1 binary labels).

### Step 3: Running the Enhanced Model (V25)
To execute the complete pipeline — which includes automatically training the Base Model, applying the CMI memory cleaning filter, executing the LARA retraining phase, and calculating anomaly scores using MAD-FES — simply run the version 25 script:

```bash
python Version25.py
```

*Note: The script will display a progress bar for each machine being processed and print the detailed F1-score for each machine immediately upon completion.*

---

## 6. Experimental Results

Evaluating version **V25** across all 28 machines of the Server Machine Dataset has yielded highly impressive aggregated performance metrics:

* **Average F1-Score:** **0.9324**
* **Median F1-Score:** **0.9740**

Notably, the model achieved near-perfect F1-scores (> 0.99) on several specific machines (e.g., `machine-1-1`, `machine-2-8`). This demonstrates the robustness and power of the MAD-FES standardization method in resisting local noise and sudden data shifts.

---

## 7. Conclusion & Future Work

**Conclusion:** The enhancements introduced in V25.0 clearly demonstrate that standardizing and cleaning the initialization memory (CMI), combined with a mutation-resistant statistical scale (MAD-FES), significantly elevates the performance of the base LARA architecture. The model not only reacts quickly to data shifts but also avoids learning incorrectly from hidden noise in the training set.

**Future Work:** Future research could focus on dynamically adjusting the window size (Dynamic Window Sizing) for the Ensemble mechanism. By automatically identifying the distinct periodicity of each feature on every machine, the model can further minimize the false positive rate in highly volatile and random data streams.
