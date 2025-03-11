"""
svg_mmnist_fd - Compute the Fréchet distance between two sets of (Stochastic) Moving MNIST images
"""

# Export the frechet_distance function and FrechetDistanceCalculator class
from svg_mmnist_fd.frechet_distance import frechet_distance, FrechetDistanceCalculator

# Package name
__name__ = "svg_mmnist_fd"

__all__ = ['frechet_distance', 'FrechetDistanceCalculator'] 