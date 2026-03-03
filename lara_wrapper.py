## lara_wrapper.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

# Assuming base_model.py defines the VAE class
from base_model import VAE


class LARA(nn.Module):
    """
    Light and Anti-overfitting Retraining Approach (LARA) wrapper.
    
    This module wraps a pre-trained VAE-based anomaly detector and introduces
    two linear adjustment layers (Mz and Mx) that facilitate fast, convex, 
    and anti-overfitting retraining for shifting data distributions.
    
    Following Theorem 1 of the LARA paper, linear formations for Mz and Mx 
    are mathematically optimal for minimizing adjustment errors.
    """

    def __init__(self, base_vae: VAE, config: dict):
        """
        Initializes the LARA wrapper with a pre-trained VAE and adjustment layers.

        Args:
            base_vae: A pre-trained VAE instance to be wrapped.
            config: Configuration dictionary containing model dimensions.
        """
        super(LARA, self).__init__()
        self.base_vae: VAE = base_vae
        
        # Extract dimensions from configuration
        model_cfg = config.get('model', {})
        self.input_dim: int = model_cfg.get('input_dim', 38)
        self.latent_dim: int = model_cfg.get('latent_dim', 16)
        
        # Mz: Adjusting function for the latent vector space
        # Implements: Mz(z) = mu_i+1 + Sigma_i+1,i * inv(Sigma_i,i) * (z - mu_i)
        self.Mz: nn.Linear = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Mx: Adjusting function for the reconstructed data samples
        # Implements: Mx(x_tilde) = mu_tilde_i+1 + Sigma_tilde_i+1,i * inv(Sigma_tilde_i,i) * (x_tilde - mu_tilde_i)
        self.Mx: nn.Linear = nn.Linear(self.input_dim, self.input_dim)

        # Initialize adjustment layers and freeze the base VAE
        self.reset_adjustments()
        self.freeze_base()

    def freeze_base(self) -> None:
        """
        Freezes all parameters of the underlying VAE.
        
        LARA only updates the lightweight adjustment layers (Mz, Mx) during 
        retraining to prevent overfitting on the limited (e.g., 1%) target data.
        """
        for param in self.base_vae.parameters():
            param.requires_grad = False

    def reset_adjustments(self) -> None:
        """
        Initializes Mz and Mx layers as Identity mappings.
        
        This ensures that before any retraining, the LARA model produces 
        outputs identical to the pre-trained base VAE.
        """
        # Initialize Mz as identity
        nn.init.eye_(self.Mz.weight)
        nn.init.zeros_(self.Mz.bias)
        
        # Initialize Mx as identity
        nn.init.eye_(self.Mx.weight)
        nn.init.zeros_(self.Mx.bias)

    def adjust_latent(self, z: Tensor) -> Tensor:
        """
        Applies the linear adjustment function Mz to the latent vector.

        Args:
            z: Latent vector of shape [Batch, Latent_Dim].

        Returns:
            Adjusted latent vector of shape [Batch, Latent_Dim].
        """
        return self.Mz(z)

    def adjust_reconstruction(self, x_tilde: Tensor) -> Tensor:
        """
        Applies the linear adjustment function Mx to reconstructed data.

        Args:
            x_tilde: Reconstructed window of shape [Batch, Window_Size, Input_Dim].

        Returns:
            Adjusted reconstruction of shape [Batch, Window_Size, Input_Dim].
        """
        # nn.Linear applies to the last dimension (Input_Dim) automatically
        return self.Mx(x_tilde)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Full forward pass of the LARA-wrapped VAE.
        
        Logic Sequence:
        1. Encode input x to latent distribution parameters.
        2. Sample latent vector z from the posterior.
        3. Adjust latent vector z via Mz.
        4. Decode adjusted latent vector to base reconstruction.
        5. Adjust reconstruction via Mx.

        Args:
            x: Input data window of shape [Batch, Window_Size, Input_Dim].

        Returns:
            adjusted_z: The latent vector after Mz transformation [Batch, Latent_Dim].
            adjusted_x: The reconstructed window after Mx transformation [Batch, Window_Size, Input_Dim].
        """
        # 1. Encoding (using the frozen base encoder)
        mu, logvar = self.base_vae.encoder(x)
        
        # 2. Latent Sampling (Reparameterization trick)
        z = self.base_vae.reparameterize(mu, logvar)
        
        # 3. Latent Space Adjustment (Mz)
        adjusted_z = self.adjust_latent(z)
        
        # 4. Decoding (using the frozen base decoder with adjusted latent)
        x_recon_base = self.base_vae.decoder(adjusted_z)
        
        # 5. Output Space Adjustment (Mx)
        adjusted_x = self.adjust_reconstruction(x_recon_base)
        
        return adjusted_z, adjusted_x

    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """
        Calculates the adjusted reconstruction error for anomaly detection.
        
        This method is used during the inference phase to compute anomaly scores.

        Args:
            x: Input data window of shape [Batch, Window_Size, Input_Dim].

        Returns:
            error: Sum of squared errors between input and adjusted reconstruction [Batch].
        """
        _, x_recon_adj = self.forward(x)
        # Calculate sum of squared errors per window
        error = torch.sum((x - x_recon_adj) ** 2, dim=(1, 2))
        return error

