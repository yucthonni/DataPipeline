from typing import Any, List
from .adapters.base import BaseAdapter
from .models.base import BaseDiffusionModel

class AugmentationPipeline:
    """
    The core pipeline orchestrator.
    It takes an adapter (to handle specific data types) and a diffusion model.
    """
    def __init__(self, adapter: BaseAdapter, model: BaseDiffusionModel):
        self.adapter = adapter
        self.model = model

    def finetune(self, data: Any, epochs: int = 10, batch_size: int = 8):
        """
        Adapts the model to a new data distribution via Few-Shot fine-tuning.
        """
        tensor_data = self.adapter.encode(data)
        self.model.finetune(tensor_data, epochs=epochs, batch_size=batch_size)

    def augment(self, data: Any, noise_level: int = 5, num_samples: int = 1) -> List[Any]:
        """
        Augment the input data by passing it through the diffusion model.
        
        Args:
            data: The input data (format depends on the adapter used).
            noise_level: The number of diffusion steps to simulate.
            num_samples: How many augmented samples to generate per input.
            
        Returns:
            A list of augmented data samples in their original format.
        """
        # 1. Encode data into standardized tensor
        tensor_data = self.adapter.encode(data)
        
        added_batch_dim = False
        # If the model is expecting a batch dimension but tensor_data is missing it:
        # MLPDenoiser expects 2D (B, D), UNetDenoiser expects 4D (B, C, H, W).
        if tensor_data.ndim == 1 or tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)
            added_batch_dim = True
        
        augmented_results = []
        for _ in range(num_samples):
            # 2. Pass tensor through the diffusion model
            augmented_tensor = self.model.augment(tensor_data, noise_level=noise_level)
            
            # 3. Remove batch dimension if we added it
            if added_batch_dim:
                augmented_tensor = augmented_tensor[0]
            
            # 4. Decode tensor back to original format
            output_data = self.adapter.decode(augmented_tensor)
            augmented_results.append(output_data)
            
        return augmented_results
