import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from .base import BaseDiffusionModel
from .networks import MLPDenoiser, UNetDenoiser

class DDPM(BaseDiffusionModel):
    """
    Standard Denoising Diffusion Probabilistic Model.
    Can be configured to use MLPDenoiser (for 1D) or UNetDenoiser (for 2D).
    """
    def __init__(self, data_shape, timesteps: int = 100, device='cpu'):
        """
        data_shape: Shape of a single data sample (e.g., (5,) for tabular, (3, 64, 64) for images).
        """
        self.device = device
        self.timesteps = timesteps
        
        # Determine network architecture based on data shape
        if len(data_shape) == 1:
            # 1D Tabular data
            self.network = MLPDenoiser(data_dim=data_shape[0]).to(self.device)
        elif len(data_shape) == 3:
            # 2D Image data (C, H, W)
            self.network = UNetDenoiser(in_channels=data_shape[0], out_channels=data_shape[0]).to(self.device)
        else:
            raise ValueError(f"Unsupported data shape: {data_shape}")

        # DDPM schedules
        self.beta = torch.linspace(0.0001, 0.02, timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Add noise to x_0 up to timestep t.
        """
        x_0 = x_0.to(self.device)
        # Create a batched t tensor
        t_tensor = torch.full((x_0.shape[0],), t, dtype=torch.long, device=self.device)
        
        noise = torch.randn_like(x_0).to(self.device)
        alpha_hat_t = self.alpha_hat[t_tensor]
        
        # Reshape alpha_hat_t to match x_0 dimensions for broadcasting
        for _ in range(len(x_0.shape) - 1):
            alpha_hat_t = alpha_hat_t.unsqueeze(-1)
            
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        return x_t

    def reverse_diffusion(self, x_t: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Denoise from x_t back to x_0 (or slightly denoised depending on steps).
        """
        self.network.eval()
        x = x_t.to(self.device)
        
        with torch.no_grad():
            for i in reversed(range(steps)):
                t = torch.full((x.shape[0],), i, dtype=torch.long, device=self.device)
                predicted_noise = self.network(x, t)
                
                alpha_t = self.alpha[t]
                alpha_hat_t = self.alpha_hat[t]
                beta_t = self.beta[t]
                
                for _ in range(len(x.shape) - 1):
                    alpha_t = alpha_t.unsqueeze(-1)
                    alpha_hat_t = alpha_hat_t.unsqueeze(-1)
                    beta_t = beta_t.unsqueeze(-1)
                
                # Equation 11 from DDPM paper
                mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise)
                
                if i > 0:
                    noise = torch.randn_like(x)
                    variance = torch.sqrt(beta_t) * noise
                    x = mean + variance
                else:
                    x = mean
                    
        return x

    def finetune(self, x: torch.Tensor, epochs: int = 10, batch_size: int = 8):
        """
        Few-shot fine-tuning on the given data x.
        """
        print(f"Starting Few-Shot Fine-tuning on {x.shape[0]} samples for {epochs} epochs...")
        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        
        dataset = TensorDataset(x.to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch_x = batch[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (batch_x.shape[0],), device=self.device).long()
                
                # Forward diffusion (add noise)
                noise = torch.randn_like(batch_x)
                alpha_hat_t = self.alpha_hat[t]
                for _ in range(len(batch_x.shape) - 1):
                    alpha_hat_t = alpha_hat_t.unsqueeze(-1)
                    
                x_t = torch.sqrt(alpha_hat_t) * batch_x + torch.sqrt(1 - alpha_hat_t) * noise
                
                # Predict noise
                optimizer.zero_grad()
                predicted_noise = self.network(x_t, t)
                
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
            
        print("Fine-tuning complete.")
