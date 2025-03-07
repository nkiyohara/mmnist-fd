"""
Test the frechet_distance function.
"""

import torch
import pytest
import numpy as np
from svg_mmnist_fd import frechet_distance

def test_frechet_distance_function():
    """Test that the frechet_distance function works correctly with dummy data."""
    # Create dummy tensors with correct Moving MNIST dimensions (batch_size x 1 x 64 x 64)
    # Using a larger batch size of 64 as requested
    images1 = torch.randn(64, 1, 64, 64)
    images2 = torch.randn(64, 1, 64, 64)
    
    # Test with CPU to ensure it works in CI environment
    try:
        # We're just testing that the function runs without errors
        # The actual value isn't important for this basic test
        fd = frechet_distance(images1, images2, device='cpu')
        
        # Check that the result is a scalar tensor, float, or numpy float type
        assert isinstance(fd, (float, torch.Tensor, np.floating))
        
        # If it's a tensor, it should be a scalar (0-dimensional)
        if isinstance(fd, torch.Tensor):
            assert fd.ndim == 0
            
        # The Fréchet distance should be non-negative
        assert float(fd) >= 0
        
    except Exception as e:
        pytest.fail(f"frechet_distance function failed with error: {e}") 