import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class Encoder(nn.Module):
    """
    Encoder component of the VAE.
    Maps input time series windows to the parameters of a Gaussian posterior distribution.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Temporal feature extraction using GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Distribution parameter heads
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape [Batch, Window_Size, Input_Dim]
            
        Returns:
            mu: Mean of the latent distribution [Batch, Latent_Dim]
            logvar: Log-variance of the latent distribution [Batch, Latent_Dim]
        """
        # GRU outputs: (output, h_n). h_n shape: [num_layers, batch, hidden_dim]
        _, h_n = self.rnn(x)
        
        # Use the last hidden state for the distribution parameters
        h_last = h_n[-1] 
        
        mu = self.mu_head(h_last)
        logvar = self.logvar_head(h_last)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder component of the VAE.
    Reconstructs the original time series window from a latent vector z.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int, window_size: int):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.window_size = window_size

        # Project latent vector to hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Temporal reconstruction using GRU
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Output mapping to original feature space
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z: Latent vector of shape [Batch, Latent_Dim]
            
        Returns:
            x_recon: Reconstructed window [Batch, Window_Size, Input_Dim]
        """
        batch_size = z.size(0)
        
        # Initial hidden state derived from latent vector
        h_0 = self.latent_to_hidden(z).unsqueeze(0) # [1, Batch, Hidden_Dim]
        
        # Provide repeated latent information as input at each time step
        # This enhances the decoder's ability to reconstruct the specific pattern
        rnn_input = h_0.transpose(0, 1).repeat(1, self.window_size, 1) # [Batch, Window_Size, Hidden_Dim]
        
        out, _ = self.rnn(rnn_input, h_0)
        
        # Map hidden states to original input dimensions
        x_recon = self.output_layer(out)
        
        return x_recon


class VAE(nn.Module):
    """
    Base Variational Auto-Encoder for Time Series Anomaly Detection.
    Implements a stochastic generative model for windowed data.
    """
    def __init__(self, config: dict):
        super(VAE, self).__init__()
        # Load parameters from config.yaml
        model_cfg = config.get('model', {})
        self.input_dim = model_cfg.get('input_dim', 38)
        self.hidden_dim = model_cfg.get('hidden_dim', 100)
        self.latent_dim = model_cfg.get('latent_dim', 16)
        self.window_size = model_cfg.get('window_size', 100)

        # Initialize sub-modules
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.input_dim, self.window_size)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample z from N(mu, var) while maintaining differentiability.
        
        Args:
            mu: Mean of the distribution [Batch, Latent_Dim]
            logvar: Log-variance of the distribution [Batch, Latent_Dim]
            
        Returns:
            z: Sampled latent vector [Batch, Latent_Dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass.
        
        Args:
            x: Input tensor [Batch, Window_Size, Input_Dim]
            
        Returns:
            x_recon: Reconstructed window [Batch, Window_Size, Input_Dim]
            mu: Latent mean [Batch, Latent_Dim]
            logvar: Latent log-variance [Batch, Latent_Dim]
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def get_reconstruction_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Computes the likelihood p(x|z) assuming a Gaussian distribution with unit variance.
        This is used by the Ruminate Block to calculate importance weights for MC sampling.
        
        Args:
            x: The observed data window [Batch, Window_Size, Input_Dim]
            z: The latent vector to evaluate [Batch, Latent_Dim]
            
        Returns:
            likelihood: The probability density p(x|z) [Batch]
        """
        # Reconstruct window from provided z
        x_recon_mu = self.decoder(z)
        
        # Calculate Gaussian log-likelihood log p(x | mu=x_recon, sigma=1)
        # Log likelihood of X given Z: -0.5 * sum((x - mu)^2 + log(2*pi))
        # We focus on the sample-wise sum of squared errors as the main component
        # Constant factors (like 2*pi) are omitted as they cancel out in normalization
        recon_error = torch.sum((x - x_recon_mu) ** 2, dim=(1, 2))
        
        # Convert log-likelihood component to likelihood (weight)
        # Note: In Monte Carlo sampling (Eq. 5), alpha_s is the product of likelihoods.
        # Returning exp(-error) represents a weight based on the reconstruction quality.
        likelihood = torch.exp(-0.5 * recon_error)
        
        return likelihood

    def reconstruction_probability(self, x: Tensor, n_samples: int = 10) -> Tensor:
        """
        Used for anomaly scoring. Calculates the average reconstruction error 
        over multiple samples from the latent posterior.
        
        Args:
            x: Input tensor [Batch, Window_Size, Input_Dim]
            n_samples: Number of MC samples to average over
            
        Returns:
            score: Reconstruction error score for each sample in batch [Batch]
        """
        mu, logvar = self.encoder(x)
        total_recon_error = torch.zeros(x.size(0), device=x.device)
        
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            total_recon_error += torch.sum((x - x_recon) ** 2, dim=(1, 2))
            
        return total_recon_error / n_samples
