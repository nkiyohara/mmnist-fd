# MMNIST-FD: Fréchet Distance Calculator for Moving MNIST

This package is a fork of [edenton/svg](https://github.com/edenton/svg) (Stochastic Video Generation with a Learned Prior), modified to provide a simple interface for calculating Fréchet distances between sets of Moving MNIST images.

## Installation

You can install this package directly from the source:

```bash
# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .

# Or build and install
pip install build
python -m build
pip install dist/mmnist-fd-0.1.0-py3-none-any.whl
```

## Usage

The package provides a simple interface for calculating the Fréchet distance between two sets of Moving MNIST images:

```python
import torch
from mmnist_fd import frechet_distance

# Load your image tensors
# images1 and images2 should be torch tensors with shape [batch_size, channels, height, width]
# For example, for Moving MNIST: [batch_size, 1, 64, 64]
images1 = torch.randn(100, 1, 64, 64)  # Replace with your actual images
images2 = torch.randn(100, 1, 64, 64)  # Replace with your actual images

# Calculate Fréchet distance
fd = frechet_distance(images1, images2)
print(f"Fréchet distance: {fd}")

# You can also specify a different model path or device
# fd = frechet_distance(images1, images2, model_path='pretrained_models/svglp_smmnist2.pth', device='cpu')
```

## Pretrained Models

The package includes the following pretrained models:

- `svglp_smmnist2.pth`: Trained on Stochastic Moving MNIST with 2 digits
- `svglp_bair.pth`: Trained on the BAIR robot push dataset

## How It Works

The Fréchet distance is calculated by:

1. Encoding the images using a pretrained encoder from the SVG-LP model
2. Computing the mean and covariance of the encoded features
3. Calculating the Fréchet distance between the two distributions

This metric is useful for evaluating the quality and diversity of generated images compared to real images.

## Citation

If you use this code in your research, please cite the original paper:

```
@inproceedings{denton2018stochastic,
  title={Stochastic Video Generation with a Learned Prior},
  author={Denton, Emily and Fergus, Rob},
  booktitle={International Conference on Machine Learning},
  year={2018}
}
```

## Acknowledgments

This package is based on the work by Emily Denton and Rob Fergus. The original repository can be found at [edenton/svg](https://github.com/edenton/svg).
