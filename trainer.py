import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from base_model import VAE
from lara_wrapper import LARA
from ruminate import RuminateBlock

class BaseTrainer:
    """
    Handles the initial unsupervised training of the base VAE model.
    Learns normal patterns from the source distribution using the Evidence Lower Bound (ELBO) loss.
    """

    def __init__(self, model: VAE, config: Dict[str, Any]):
        """
        Initializes the BaseTrainer with the VAE model and training configuration.

        Args:
            model: The VAE model instance to train.
            config: Configuration dictionary containing training hyperparameters.
        """
        self.model: VAE = model
        self.config: Dict[str, Any] = config
        
        # Training parameters from config
        train_cfg = config.get('training', {})
        self.lr: float = train_cfg.get('learning_rate', 0.001)
        self.epochs: int = train_cfg.get('epochs_base', 50)
        self.batch_size: int = train_cfg.get('batch_size', 100)
        
        # Optimizer
        self.optimizer: optim.Optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr
        )
        
        # Move model to appropriate device
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the VAE loss (Reconstruction Loss + KL Divergence).

        Args:
            x: Original input batch [Batch, Window_Size, Input_Dim].
            x_recon: Reconstructed batch [Batch, Window_Size, Input_Dim].
            mu: Latent mean [Batch, Latent_Dim].
            logvar: Latent log-variance [Batch, Latent_Dim].

        Returns:
            total_loss: The scalar loss tensor.
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Batch-averaged loss
        return (recon_loss + kl_loss) / x.size(0)

    def train(self, train_loader: DataLoader) -> None:
        """
        Executes the training loop for the base VAE.

        Args:
            train_loader: DataLoader providing windowed source distribution data.
        """
        self.model.train()
        print(f"Starting VAE pre-training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x)
                
                loss = self._vae_loss(x, x_recon, mu, logvar)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Average ELBO Loss: {avg_loss:.4f}")


class LARARetrainer:
    """
    Implements the Light and Anti-overfitting Retraining Approach.
    Optimizes the linear adjustment layers (Mx and Mz) of the LARA wrapper 
    using a small slice of target distribution data.
    """

    def __init__(self, lara: LARA, ruminate: RuminateBlock, config: Dict[str, Any]):
        """
        Initializes the LARARetrainer.

        Args:
            lara: The LARA wrapper instance containing frozen base_vae and trainable Mx, Mz.
            ruminate: The RuminateBlock instance used to estimate target latent vectors.
            config: Configuration dictionary containing retraining parameters.
        """
        self.lara: LARA = lara
        self.ruminate: RuminateBlock = ruminate
        
        # Retraining parameters from config
        train_cfg = config.get('training', {})
        self.lr: float = train_cfg.get('learning_rate', 0.001)
        self.epochs: int = train_cfg.get('epochs_retrain', 20)
        
        # Optimizer: ONLY targets parameters in Mx and Mz adjustment layers.
        # This ensures the problem remains convex and prevents overfitting.
        self.optimizer: optim.Optimizer = optim.Adam([
            {'params': self.lara.Mz.parameters()},
            {'params': self.lara.Mx.parameters()}
        ], lr=self.lr)
        
        # Loss function for convex optimization
        self.criterion: nn.MSELoss = nn.MSELoss()
        
        # Device management
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lara.to(self.device)

    def retrain(self, target_loader: DataLoader) -> None:
        """
        Executes the LARA retraining process on the provided small target dataset.
        Adapts the model to new distributions with high data efficiency (e.g., 1%).

        Args:
            target_loader: DataLoader containing the 1% slice of the target distribution.
        """
        # Ensure base model is frozen and LARA is in training mode for adjustment layers
        self.lara.freeze_base()
        self.lara.train()
        
        print(f"Starting LARA convex retraining for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, (x_target, _) in enumerate(target_loader):
                x_target = x_target.to(self.device)
                
                # 1. Estimate target latent vector (Z_tilde) via RuminateBlock
                # This leverages historical knowledge and current observation
                z_tilde_target = self.ruminate.estimate_target_z(self.lara, x_target)
                
                self.optimizer.zero_grad()
                
                # 2. Forward pass through LARA to get adjusted latent and reconstruction
                # adjusted_z = Mz(Encoder(x))
                # adjusted_x = Mx(Decoder(adjusted_z))
                adj_z, adj_x = self.lara(x_target)
                
                # 3. Compute Convex Loss (Section 3.4, Equation 8)
                # Loss = MSE(adjusted_x, x_target) + MSE(adjusted_z, z_tilde)
                loss_x = self.criterion(adj_x, x_target)
                loss_z = self.criterion(adj_z, z_tilde_target)
                
                total_loss_batch = loss_x + loss_z
                
                # 4. Update ONLY Mx and Mz parameters
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                
            avg_loss = total_loss / len(target_loader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Retrain Epoch [{epoch+1}/{self.epochs}], Convex Loss: {avg_loss:.6f}")

        print("LARA retraining complete. Identity adjustments successfully adapted to target distribution.")
