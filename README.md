LARA-TSAD: Light and Anti-overfitting Retraining Approach for Time-Series Anomaly DetectionThis repository contains the official PyTorch implementation of LARA-TSAD, a robust Test-Time Adaptation (TTA) framework designed for unsupervised Time-Series Anomaly Detection under severe distribution shifts.AbstractTime-series anomaly detection (TSAD) in real-world cyber-physical systems is often severely compromised by non-stationary environments, such as sensor drift or operational state transitions. Traditional static models suffer from catastrophic performance degradation when confronted with these distribution shifts. We propose LARA (Light and Anti-overfitting Retraining Approach), a mathematically principled framework that wraps around pre-trained generative models (e.g., Variational Autoencoders) to enable real-time adaptation without catastrophic forgetting.Version 25.0 introduces three breakthrough mechanisms: Clean Memory Initialization (CMI), MAD-FES Profiling, and Micro-Retraining, pushing the state-of-the-art boundaries on the Server Machine Dataset (SMD) to an Average F1-Score of 0.9324.Repository StructureLARA-TSAD/
├── data/                   # Directory for datasets (e.g., SMD, MSL, SMAP)
├── base_model.py           # Core VAE architecture (Encoder-Decoder)
├── config.yaml             # Centralized hyperparameter configuration
├── data_utils.py           # Memory-efficient sliding window data loaders
├── detector.py             # EVT/POT thresholding and evaluation metrics
├── lara_wrapper.py         # The LARA adaptation modules (Mz, Mx)
├── main.py                 # Main experimental pipeline orchestration
├── ruminate.py             # Associative memory module via Monte Carlo estimation
└── trainer.py              # Base ELBO optimization and Convex adaptation logic
Highlights of Version 25.0Robustness against Model Poisoning: Utilizes a restricted 1% adaptation buffer to prevent anomalies from corrupting the adaptive process.Convex Optimization: Adapts exclusively via low-dimensional affine transformations ($M_z$, $M_x$), guaranteeing fast and stable convergence during test-time.Instantaneous Fault Localization: Replaces window-averaged error formulations with Final Error Scoring (FES) to preserve the temporal resolution of sudden anomalies.Getting StartedThis guide provides step-by-step instructions to set up the environment, download necessary datasets, and reproduce the benchmark results using the Server Machine Dataset (SMD).1. Environment SetupThe codebase is built on Python 3.8+ and PyTorch. We recommend using a virtual environment (e.g., Anaconda or Python venv).# Clone the repository
git clone [https://github.com/your-username/LARA-TSAD.git](https://github.com/your-username/LARA-TSAD.git)
cd LARA-TSAD

# Create virtual environment
python -m venv lara_env
source lara_env/bin/activate  # On Windows use: lara_env\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn scipy pyyaml tqdm
2. Dataset Preparation (SMD)The current implementation natively supports the Server Machine Dataset (SMD), which consists of 28 distinct machine entities.Ensure your directory structure matches the following:data/
└── SMD/
    ├── train/          # e.g., machine-1-1.txt, machine-1-2.txt...
    ├── test/           # e.g., machine-1-1.txt, machine-1-2.txt...
    └── test_label/     # e.g., machine-1-1.txt, machine-1-2.txt...
The data_utils.py module is designed to automatically ingest these files, applying deterministic Min-Max scaling and efficient sliding-window transformations via torch.as_strided.3. Configuration ManagementAll hyperparameters are controlled centrally via config.yaml. Before running, review the core parameters:# config.yaml (Snippet)
model:
  input_dim: 38         # Standard for SMD features
  latent_dim: 32        # VAE bottleneck size
  window_size: 50       # Sliding window context

training:
  batch_size: 100
  learning_rate: 0.001
  epochs_base: 50       # Base VAE pre-training epochs
  epochs_retrain: 10    # LARA adaptation epochs
  
lara:
  n_restored: 10        # Historical samples retrieved
  mc_samples: 10        # Monte Carlo estimation samples
  retrain_ratio: 0.01   # Micro-retraining buffer (1%)
4. Execution PipelineTo execute the end-to-end evaluation protocol across all 28 SMD machines, simply run:python main.py
Workflow orchestrated by main.py:Data Loading: Initializes DatasetManager to parse and stride the time-series.Prior Learning: Instantiates VAE and utilizes BaseTrainer to learn the source normal distribution $\mathcal{D}_S$.Adaptation Phase: Wraps the trained VAE in the LARA module and executes the Micro-Retraining protocol using the first 1% of the target data via LARARetrainer.Detection & Scoring: Passes adapted scores to AnomalyDetector for EVT/POT thresholding and outputs rigorous Point-Adjusted Precision, Recall, and F1-scores.Methodological FrameworkThis section outlines the core theoretical and mathematical foundations of the LARA framework, specifically detailing the structural mechanisms introduced in Version 25.0.1. The Core LARA ArchitectureLARA operates as a Test-Time Adaptation (TTA) wrapper over a pre-trained base generative model, specifically a Variational Autoencoder (VAE). The objective is to mitigate the discrepancy between the source distribution $\mathcal{D}_S$ and the target (shifted) distribution $\mathcal{D}_T$.Instead of fine-tuning the entire high-dimensional parameter space of the VAE (which inevitably leads to catastrophic forgetting and model poisoning), LARA freezes the base parameters $\Theta_{base}$ and learns two lightweight affine transformations:Latent Adjustment Layer ($M_z$): Realigns the encoded posterior distribution.Output Adjustment Layer ($M_x$): Calibrates the decoded reconstruction to the target spatial manifold.The forward pass during adaptation is defined as:$$\tilde{z} = M_z(E_{\theta}(x))$$$$\tilde{x} = M_x(D_{\phi}(\tilde{z}))$$where $E_{\theta}$ and $D_{\phi}$ represent the frozen Encoder and Decoder, respectively.2. Version 25.0 InnovationsThe exceptional empirical performance of V25.0 stems from three synergistic components designed to ensure stability under extreme anomaly contamination.2.1 Clean Memory Initialization (CMI)The RuminateBlock relies on historical data to estimate a stable target latent vector $\tilde{Z}$ for incoming streams via Monte Carlo expectation. However, random sampling from the historical buffer $\mathcal{H}$ may retrieve anomalous instances, corrupting the target estimation.CMI Protocol:Prior to deployment, the historical buffer $\mathcal{H}$ is evaluated using the base VAE's log-likelihood objective. We define a reconstruction error set $\mathcal{E}$. The buffer is pruned based on a conservative quantile threshold $\tau_{prune} = 0.90$:$$\mathcal{H}_{clean} = \{ x_i \in \mathcal{H} \mid \text{Error}(x_i) \leq \text{Quantile}(\mathcal{E}, \tau_{prune}) \}$$This guarantees that the associative memory mechanism references a purified normative prior, producing unbiased gradient signals for $M_z$ and $M_x$.2.2 Micro-Retraining (The 1% Buffer Constraint)Traditional unsupervised TTA approaches often assume monotonicity between adaptation performance and buffer size. We challenge this assumption in anomaly detection. A larger target buffer $B_T$ monotonically increases the probability of ingesting out-of-distribution anomalies into the retraining loop, causing the model to interpret anomalies as the "new normal" (Model Poisoning).Micro-Retraining Protocol:We constrain the adaptation buffer to exactly $1\%$ of the initial test stream ($\tau_{retrain} = 0.01$).Because the learnable parameters ($M_z$, $M_x$) define convex, linear transformations, the optimization landscape requires minimal data points to locate the global minimum. This acts as an implicit regularizer: the limited buffer provides sufficient distributional variance to perform an axis-shift alignment, while lacking the requisite entropy to overfit to localized anomalies.2.3 Robust MAD-FES ProfilingThe final anomaly score $S(t)$ dictates the detection efficacy. Previous iterations relied on window-averaged Mean Squared Error (MSE) normalized via standard Gaussian moments ($\mu, \sigma$).Final Error Scoring (FES):Averaging over a sliding window $W$ of size $L$ causes error dilution, masking point-anomalies. FES extracts the reconstruction error strictly at the terminal timestep of the window:$$\text{FES}(x_t) = \| x_t - \tilde{x}_t \|_2^2 \quad \text{where } t \text{ is the final point in } W$$Median Absolute Deviation (MAD) Normalization:Standard normalization is highly vulnerable to extreme outliers, leading to threshold inflation. MAD is a robust rank-statistic that remains stable even when up to 50% of the distribution is corrupted.The thresholding logic within our Extreme Value Theory (EVT) Peak-Over-Threshold (POT) module is pre-conditioned using:$$S_{norm}(t) = \frac{\text{FES}(t) - \text{Median}(\text{FES})}{\text{MAD}(\text{FES})}$$$$\text{MAD}(\text{FES}) = \text{Median}( | \text{FES} - \text{Median}(\text{FES}) | )$$This dual-pronged approach yields an anomaly threshold that is intensely sensitive to instantaneous deviations yet impervious to long-term noise distortion.CitationIf you find this repository useful in your research, please consider citing our work:@article{lara_tsad_2026,
  title={LARA: Light and Anti-overfitting Retraining Approach for Unsupervised Time-Series Anomaly Detection},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
