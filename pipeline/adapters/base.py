from abc import ABC, abstractmethod
import torch
from typing import Any

class BaseAdapter(ABC):
    """
    Base adapter to convert external data into a standardized tensor format 
    suitable for the diffusion model, and vice versa.
    """
    
    @abstractmethod
    def encode(self, data: Any) -> torch.Tensor:
        """
        Convert the raw data into a PyTorch tensor.
        """
        pass

    @abstractmethod
    def decode(self, tensor: torch.Tensor) -> Any:
        """
        Convert the PyTorch tensor back into the raw data format.
        """
        pass
