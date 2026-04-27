from abc import ABC, abstractmethod
import torch

class BaseDiffusionModel(ABC):
    """
    Base class for diffusion models inside the augmentation pipeline.
    It encapsulates the forward diffusion (adding noise) and 
    reverse diffusion (denoising/generation) processes.
    """

    @abstractmethod
    def forward_diffusion(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Adds noise to the data `x` up to timestep `t`.
        """
        pass

    @abstractmethod
    def reverse_diffusion(self, x_t: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Denoises the data from timestep `x_t` for `steps`.
        """
        pass

    def finetune(self, x: torch.Tensor, epochs: int = 10, batch_size: int = 32):
        """
        Few-shot fine-tuning on the given data `x` to adapt to new distributions.
        By default, does nothing unless implemented.
        """
        pass

    def augment(self, x: torch.Tensor, noise_level: int) -> torch.Tensor:
        """
        Convenience method that performs forward diffusion up to `noise_level` 
        and then reverse diffusion back to generate augmented data.
        """
        # Add noise
        noisy_x = self.forward_diffusion(x, t=noise_level)
        # Reconstruct/Generate
        augmented_x = self.reverse_diffusion(noisy_x, steps=noise_level)
        return augmented_x
