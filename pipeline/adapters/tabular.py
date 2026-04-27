from .base import BaseAdapter
import torch
import numpy as np

class TabularAdapter(BaseAdapter):
    """
    Adapter for tabular data.
    Converts 1D or 2D tabular features into tensors and optionally normalizes them.
    """
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        # In a real scenario, we might want to store mean/std for denormalization
        self.mean = 0.0
        self.std = 1.0

    def encode(self, data: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(data).float()
        
        if self.normalize:
            self.mean = tensor.mean(dim=0)
            self.std = tensor.std(dim=0) + 1e-8
            tensor = (tensor - self.mean) / self.std
            
        return tensor

    def decode(self, tensor: torch.Tensor) -> np.ndarray:
        if self.normalize:
            tensor = tensor * self.std + self.mean
            
        return tensor.detach().cpu().numpy()
