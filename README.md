# DataPipeline

A flexible, modular framework for data augmentation using generative models like **DDPM** (Denoising Diffusion Probabilistic Models). Designed to support both tabular data and image processing pipelines, it includes few-shot fine-tuning capabilities to adapt models to target distributions.

## Features

- **DDPM Core**: Modular implementation of Denoising Diffusion Probabilistic Models.
- **Data Adapters**: Easily switch between `TabularAdapter` and `ImageAdapter`.
- **Few-Shot Fine-tuning**: Adapt pre-trained generative models to new distributions with minimal data.
- **Unified Pipeline**: A consistent API (`augment`, `finetune`) for different data modalities.

## Project Structure

```text
/pipeline
├── /adapters    # Logic for data preparation (Normalization, etc.)
├── /models      # Model definitions (DDPM, Networks)
├── core.py      # Main AugmentationPipeline logic
└── main.py      # Entry point for testing and usage
```

## Quick Start

### Prerequisites
- Python 3.14+
- `torch`, `numpy`, `fsspec`, `jinja2`, `networkx`, `sympy` (see `requirements.txt`)

### Run Tests
Execute the provided test suite to verify the installation:

```bash
python main.py
```

## Usage Example

```python
from pipeline import AugmentationPipeline, TabularAdapter, DDPM

# Setup Pipeline
adapter = TabularAdapter(normalize=True)
model = DDPM(data_shape=(5,), timesteps=50) 
pipeline = AugmentationPipeline(adapter=adapter, model=model)

# Fine-tune on target data
pipeline.finetune(target_data, epochs=10)

# Augment
augmented_data = pipeline.augment(seed_data, num_samples=5)
```

## License
MIT
