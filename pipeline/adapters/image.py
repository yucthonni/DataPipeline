from .base import BaseAdapter
import torch
import numpy as np

class ImageAdapter(BaseAdapter):
    """
    Adapter for image data. 
    Expects input data as numpy arrays (e.g., shape HWC or CHW) and converts
    them to standard PyTorch tensors (NCHW or CHW).
    """
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def encode(self, data: np.ndarray) -> torch.Tensor:
        # Convert numpy array to tensor
        tensor = torch.from_numpy(data).float()
        
        # Simple heuristic to ensure channel-first (CHW) if it seems like HWC
        if tensor.ndim == 3 and tensor.shape[-1] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)
            
        if self.normalize:
            # Assuming 8-bit image
            tensor = tensor / 255.0 * 2.0 - 1.0 # Scale to [-1, 1]
            
        return tensor

    def decode(self, tensor: torch.Tensor) -> np.ndarray:
        if self.normalize:
            tensor = (tensor + 1.0) / 2.0 * 255.0
            tensor = torch.clamp(tensor, 0, 255)
            
        # Convert back to numpy
        data = tensor.detach().cpu().numpy().astype(np.uint8)
        
        # Convert back to HWC for standard image representation
        if data.ndim == 3 and data.shape[0] in [1, 3, 4]:
            data = np.transpose(data, (1, 2, 0))
            
        return data
