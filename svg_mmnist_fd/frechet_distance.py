import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy import linalg
from svg_mmnist_fd import utils
from svg_mmnist_fd import models
import importlib.resources
import pathlib
import sys

# Global cache for loaded models
_MODEL_CACHE = {}

# Function to get the path to a resource file
def get_resource_path(resource_name):
    """
    Get the path to a resource file in the package.
    
    Args:
        resource_name (str): Name of the resource file
        
    Returns:
        str: Path to the resource file
    """
    # First try to find the file in the package resources
    try:
        # For Python 3.9+ (compatible with Python 3.13+)
        pkg_dir = importlib.resources.files('svg_mmnist_fd')
        resource_path = pkg_dir / resource_name
        if resource_path.exists():
            return str(resource_path)
    except (ImportError, AttributeError):
        # Fallback for older Python versions
        try:
            return str(importlib.resources.path('svg_mmnist_fd', resource_name))
        except (ImportError, FileNotFoundError):
            pass
    
    # If not found in package resources, try relative path
    if os.path.exists(resource_name):
        return resource_name
    
    # If still not found, return the original path and hope for the best
    return resource_name


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two multivariate Gaussians.
    
    Args:
        mu1 (torch.Tensor): Mean of first distribution
        sigma1 (torch.Tensor): Covariance matrix of first distribution
        mu2 (torch.Tensor): Mean of second distribution
        sigma2 (torch.Tensor): Covariance matrix of second distribution
        eps (float): Small constant for numerical stability
        
    Returns:
        float: Fréchet Distance
    """
    mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
    sigma1, sigma2 = sigma1.cpu().numpy(), sigma2.cpu().numpy()
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def load_mmnist_model(model_path='pretrained_models/svglp_smmnist2.pth', device='cuda'):
    """
    Load the pretrained MMNIST model.
    
    Args:
        model_path (str): Path to the pretrained model
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (encoder, frame_predictor, posterior, prior, decoder, opt)
    """
    # Check if model is already in cache
    cache_key = f"{model_path}_{device}"
    if cache_key in _MODEL_CACHE:
        print(f"Using cached model for {model_path}")
        return _MODEL_CACHE[cache_key]
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, using CPU instead.")
        device = 'cpu'
    
    # 古いモジュール構造を新しい構造にマッピング
    # 'models'モジュールを'svg_mmnist_fd.models'にマッピング
    sys.modules['models'] = sys.modules['svg_mmnist_fd.models']
    
    # Load the pretrained model with weights_only=False to handle older model files
    saved_model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    
    # マッピングを削除（オプション）
    # del sys.modules['models']
    
    # Extract model components
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
    encoder = saved_model['encoder']
    decoder = saved_model['decoder']
    opt = saved_model['opt']
    
    # Move models to the specified device
    frame_predictor.to(device)
    posterior.to(device)
    prior.to(device)
    encoder.to(device)
    decoder.to(device)
    
    # Set models to evaluation mode
    frame_predictor.eval()
    posterior.eval()
    prior.eval()
    encoder.eval()
    decoder.eval()
    
    # Cache the loaded model
    _MODEL_CACHE[cache_key] = (encoder, frame_predictor, posterior, prior, decoder, opt)
    
    return encoder, frame_predictor, posterior, prior, decoder, opt


def preprocess_images(images, opt=None, device='cuda'):
    """
    Preprocess images to be compatible with the MMNIST model.
    
    Args:
        images (torch.Tensor): Images to preprocess, shape [batch_size, channels, height, width]
        opt (object): Options for preprocessing
        device (str): Device to move the images to
        
    Returns:
        torch.Tensor: Preprocessed images
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    
    # If images are in range [0, 255], normalize to [0, 1]
    if images.max() > 1.0:
        images = images / 255.0
    
    # If opt is provided, ensure images match the expected format
    if opt is not None:
        # Resize images if needed
        if images.shape[2] != opt.image_width or images.shape[3] != opt.image_width:
            images = F.interpolate(
                images, 
                size=(opt.image_width, opt.image_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Ensure correct number of channels
        if images.shape[1] != opt.channels:
            if opt.channels == 1 and images.shape[1] == 3:
                # Convert RGB to grayscale
                images = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            elif opt.channels == 3 and images.shape[1] == 1:
                # Convert grayscale to RGB
                images = images.repeat(1, 3, 1, 1)
    
    return images.to(device)


def encode_images(images, encoder, device='cuda'):
    """
    Encode images using the encoder part of the MMNIST model.
    
    Args:
        images (torch.Tensor): Images to encode, shape [batch_size, channels, height, width]
        encoder (nn.Module): Encoder model
        device (str): Device to perform encoding on
        
    Returns:
        torch.Tensor: Encoded features, shape [batch_size, feature_dim]
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    
    images = images.to(device)
    
    with torch.no_grad():
        features, _ = encoder(images)
    
    return features


def calculate_statistics(features):
    """
    Calculate mean and covariance statistics of features.
    
    Args:
        features (torch.Tensor): Features to calculate statistics for, shape [batch_size, feature_dim]
        
    Returns:
        tuple: (mean, covariance)
    """
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.t())
    
    return mu, sigma


def frechet_distance(images1, images2, model_path='pretrained_models/svglp_smmnist2.pth', device='cuda'):
    """
    Calculate the Fréchet distance between two sets of images using the MMNIST model's encoder.
    
    Args:
        images1 (torch.Tensor): First set of images, shape [batch_size, channels, height, width]
        images2 (torch.Tensor): Second set of images, shape [batch_size, channels, height, width]
        model_path (str): Path to the pretrained model
        device (str): Device to perform calculations on ('cuda' or 'cpu')
        
    Returns:
        float: Fréchet distance between the two sets of images
    """
    # Get the actual path to the model file
    actual_model_path = get_resource_path(model_path)
    
    # Load the model
    encoder, _, _, _, _, opt = load_mmnist_model(actual_model_path, device)
    
    # Preprocess images
    images1 = preprocess_images(images1, opt, device)
    images2 = preprocess_images(images2, opt, device)
    
    # Encode images
    features1 = encode_images(images1, encoder, device)
    features2 = encode_images(images2, encoder, device)
    
    # Calculate statistics
    mu1, sigma1 = calculate_statistics(features1)
    mu2, sigma2 = calculate_statistics(features2)
    
    # Calculate Fréchet distance
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid


def clear_model_cache():
    """
    Clear the model cache to free up memory.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    print("Model cache cleared.")


if __name__ == "__main__":
    # Example usage
    # Generate random images for demonstration
    batch_size = 100
    channels = 1
    height = 64
    width = 64
    
    # Random images in range [0, 1]
    images1 = torch.rand(batch_size, channels, height, width)
    images2 = torch.rand(batch_size, channels, height, width)
    
    # Calculate Fréchet distance
    distance = frechet_distance(images1, images2, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fréchet distance: {distance}")
    
    # Generate another set of images and calculate distance again (should use cached model)
    images3 = torch.rand(batch_size, channels, height, width)
    distance2 = frechet_distance(images1, images3, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fréchet distance (second call): {distance2}")
    
    # Clear cache if needed
    # clear_model_cache() 