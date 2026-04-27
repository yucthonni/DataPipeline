import torch
import numpy as np
from pipeline import AugmentationPipeline, ImageAdapter, TabularAdapter, DDPM

def test_tabular_fewshot():
    print("=== Testing Tabular Few-Shot with DDPM ===")
    
    # 1. Base Domain Data (e.g., mean ~50)
    base_table = np.random.normal(loc=50.0, scale=5.0, size=(10, 5))
    print(f"Base data shape: {base_table.shape}, mean value: {base_table.mean():.2f}")

    # 2. Setup Pipeline with DDPM
    adapter = TabularAdapter(normalize=True)
    # Determine shape: it expects a single sample shape
    # The adapter encodes it into [N, 5]. A single sample is [5].
    model = DDPM(data_shape=(5,), timesteps=50) 
    pipeline = AugmentationPipeline(adapter=adapter, model=model)

    # 3. Target Domain Data (e.g., mean ~1000 - a huge distribution shift)
    target_table = np.random.normal(loc=1000.0, scale=10.0, size=(10, 5))
    print(f"Target distribution mean value: {target_table.mean():.2f}")

    # 4. Few-Shot Fine-Tuning
    # The pipeline will adjust the model weights to the target distribution
    pipeline.finetune(target_table, epochs=10, batch_size=4)

    # 5. Augment new samples based on the new distribution
    # We pass in some dummy seeds from the new distribution or just noise
    augmented_tables = pipeline.augment(target_table, noise_level=10, num_samples=1)
    
    for i, tab in enumerate(augmented_tables):
        print(f"Augmented table {i+1} shape: {tab.shape}, mean value: {tab.mean():.2f}")
        
    print("Tabular Few-Shot works successfully!\n")

def test_image_ddpm():
    print("=== Testing Image Pipeline with DDPM ===")
    
    # Image requires UNetDenoiser. Input is (C, H, W)
    dummy_image = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    print(f"Original image shape: {dummy_image.shape}, mean pixel value: {dummy_image.mean():.2f}")

    adapter = ImageAdapter(normalize=True)
    model = DDPM(data_shape=(3, 16, 16), timesteps=20)
    pipeline = AugmentationPipeline(adapter=adapter, model=model)

    # Augment
    augmented_images = pipeline.augment(dummy_image, noise_level=5, num_samples=1)

    for i, img in enumerate(augmented_images):
        print(f"Augmented image {i+1} shape: {img.shape}, mean pixel value: {img.mean():.2f}")
    
    print("Image DDPM pipeline works successfully!\n")

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_tabular_fewshot()
    test_image_ddpm()
