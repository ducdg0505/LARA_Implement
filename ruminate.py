## ruminate.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class RuminateBlock:
    """
    Ruminate Block for LARA.
    
    This component leverages the 'historical knowledge' embedded in a frozen VAE 
    to guide the fine-tuning of the latent vector space for new distributions.
    It estimates a target latent vector (Z_tilde) for a new data sample by 
    calculating a weighted expectation using Monte Carlo sampling.
    """

    def __init__(self, config: dict):
        """
        Initializes the RuminateBlock with parameters from config.yaml.

        Args:
            config: Configuration dictionary containing lara and model parameters.
        """
        lara_cfg = config.get('lara', {})
        model_cfg = config.get('model', {})
        
        # n: Number of restored historical samples per observation
        self.n_samples: int = lara_cfg.get('n_restored', 3)
        
        # N: Number of samples for Monte Carlo estimation
        self.mc_samples: int = lara_cfg.get('mc_samples', 10)
        
        # Dimensionality of the latent space
        self.latent_dim: int = model_cfg.get('latent_dim', 16)

    @torch.no_grad()
    def estimate_target_z(self, lara: nn.Module, x_new: Tensor) -> Tensor:
        """
        Estimates the target latent vector (Z_tilde) for the current observation.
        
        Following Section 3.2 (Equation 3 & 5) of the paper, this involves:
        1. Generating n restored historical samples from the frozen base model.
        2. Sampling N candidate latent vectors from a standard normal prior.
        3. Calculating likelihood weights based on reconstruction quality.
        4. Computing the weighted expectation of the candidates.

        Args:
            lara: The LARA wrapper instance containing the frozen base_vae.
            x_new: Newly observed data batch of shape [Batch, Window_Size, Input_Dim].

        Returns:
            z_tilde: The estimated target latent vector [Batch, Latent_Dim].
        """
        batch_size = x_new.size(0)
        device = x_new.device
        base_vae = lara.base_vae

        # --- Step 1: Generate Restored Historical Samples (X_bar) ---
        # Get the latent parameters from the frozen encoder for current data
        mu_old, logvar_old = base_vae.encoder(x_new)
        
        # Restore n samples representing 'historical knowledge' for this pattern
        # This acts as a proxy for historical data similar to the current input
        restored_samples = []
        for _ in range(self.n_samples):
            z_hist = base_vae.reparameterize(mu_old, logvar_old)
            x_bar = base_vae.decoder(z_hist)
            restored_samples.append(x_bar) # Each is [Batch, Window_Size, Input_Dim]

        # --- Step 2: Monte Carlo Prior Sampling (Z_s) ---
        # Sample N candidate vectors from the prior p(z) = N(0, I)
        # Shape: [Batch, N, Latent_Dim]
        z_candidates = torch.randn(batch_size, self.mc_samples, self.latent_dim, device=device)

        # --- Step 3: Calculate Likelihood Weights (alpha_s) ---
        # Use log-likelihood for numerical stability
        # Log likelihood logic is derived from p(x|z) calculation in base_model.py
        total_log_likelihood = torch.zeros(batch_size, self.mc_samples, device=device)

        for s in range(self.mc_samples):
            # Extract current MC candidate latent vector for the whole batch
            z_s = z_candidates[:, s, :] # [Batch, Latent_Dim]
            
            # Log Likelihood of the new observation: log p(x_new | z_s)
            # base_vae.get_reconstruction_likelihood returns exp(-0.5 * error)
            # We add a small epsilon 1e-9 to avoid log(0)
            p_new = base_vae.get_reconstruction_likelihood(x_new, z_s)
            log_p_new = torch.log(p_new + 1e-9)
            
            # Log Likelihood of the restored historical samples: sum(log p(x_bar | z_s))
            log_p_hist = torch.zeros(batch_size, device=device)
            for j in range(self.n_samples):
                p_hist = base_vae.get_reconstruction_likelihood(restored_samples[j], z_s)
                log_p_hist += torch.log(p_hist + 1e-9)
                
            # Total importance weight for this MC sample: log(alpha_s)
            total_log_likelihood[:, s] = log_p_new + log_p_hist

        # --- Step 4: Weighted Expectation (Eq. 5) ---
        # Use Softmax over the N samples dimension to normalize alpha_s weights
        # Softmax effectively converts log-likelihoods to importance weights
        weights = torch.softmax(total_log_likelihood, dim=1) # [Batch, N]
        
        # Compute the expected latent vector: E[z] = sum(weight_s * z_s)
        # Expand weights for broadcasting with candidates: [Batch, N, 1] * [Batch, N, Latent_Dim]
        z_tilde = torch.sum(weights.unsqueeze(-1) * z_candidates, dim=1) # [Batch, Latent_Dim]

        return z_tilde
