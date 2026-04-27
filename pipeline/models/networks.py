import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MLPDenoiser(nn.Module):
    """
    A simple MLP for denoising 1D data (e.g. tabular data).
    """
    def __init__(self, data_dim: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        h = self.fc1(x)
        h = h + t_emb # Inject time
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        out = self.fc3(h)
        return out


class UNetDenoiser(nn.Module):
    """
    A minimal UNet for denoising 2D data (e.g. images).
    Assumes square inputs where H, W are reasonably small (like 64x64).
    """
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, base_channels),
            nn.SiLU()
        )
        
        # Down
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        
        # Mid
        self.mid1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        self.mid2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        
        # Up
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(base_channels * 2, out_channels, 3, padding=1)
        
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # Expand t_emb to match image spatial dims
        t_emb_spatial = t_emb[:, :, None, None]
        
        # Down
        h1 = self.act(self.conv1(x)) + t_emb_spatial
        h2 = self.act(self.conv2(h1))
        
        # Mid
        h_mid = self.act(self.mid1(h2))
        h_mid = self.act(self.mid2(h_mid))
        
        # Up
        h3 = self.act(self.up1(h_mid))
        # Skip connection
        h3_cat = torch.cat([h3, h1], dim=1)
        
        out = self.conv_out(h3_cat)
        return out
