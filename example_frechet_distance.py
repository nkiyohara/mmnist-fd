import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mmnist_fd.frechet_distance import frechet_distance
from mmnist_fd.data.moving_mnist import MovingMNIST

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Moving MNIST dataset
    data_root = './data'
    seq_len = 20  # Number of frames in each sequence
    num_digits = 2  # Number of digits in each sequence
    image_size = 64  # Size of each frame
    
    # Load the Moving MNIST test dataset
    print("Loading Moving MNIST dataset...")
    test_data = MovingMNIST(
        train=False,
        data_root=data_root,
        seq_len=seq_len,
        num_digits=num_digits,
        image_size=image_size,
        deterministic=False
    )
    
    # Create a dataloader
    batch_size = 100
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Get a batch of sequences
    sequences = next(iter(test_loader))  # Shape: [batch_size, seq_len, height, width, channels]
    
    # Extract frames from different parts of the sequences
    # Set 1: First frames of each sequence
    # Set 2: Last frames of each sequence
    images_set1 = sequences[:, 0]  # Shape: [batch_size, height, width, channels]
    images_set2 = sequences[:, -1]  # Shape: [batch_size, height, width, channels]
    
    # Convert to the format expected by the model: [batch_size, channels, height, width]
    images_set1 = images_set1.permute(0, 3, 1, 2)
    images_set2 = images_set2.permute(0, 3, 1, 2)
    
    print(f"Set 1 shape: {images_set1.shape}")
    print(f"Set 2 shape: {images_set2.shape}")
    
    # Calculate Fréchet distance between the two sets
    fd = frechet_distance(
        images_set1, 
        images_set2, 
        model_path='pretrained_models/svglp_smmnist2.pth',
        device=device
    )
    
    print(f"Fréchet distance between first and last frames: {fd}")
    
    # Now let's create a more similar set by adding noise to set1
    noise_level = 0.1
    noisy_images_set1 = images_set1 + noise_level * torch.randn_like(images_set1)
    noisy_images_set1 = torch.clamp(noisy_images_set1, 0, 1)
    
    # Calculate Fréchet distance between original set1 and noisy set1
    fd_noise = frechet_distance(
        images_set1, 
        noisy_images_set1, 
        model_path='pretrained_models/svglp_smmnist2.pth',
        device=device
    )
    
    print(f"Fréchet distance between original and noisy first frames (noise level {noise_level}): {fd_noise}")
    
    # Visualize some examples
    plt.figure(figsize=(12, 6))
    
    # Original images from set 1 (first frames)
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images_set1[i, 0].numpy(), cmap='gray')
        plt.title(f"First Frame - {i}")
        plt.axis('off')
    
    # Images from set 2 (last frames)
    for i in range(5):
        plt.subplot(3, 5, i + 6)
        plt.imshow(images_set2[i, 0].numpy(), cmap='gray')
        plt.title(f"Last Frame - {i}")
        plt.axis('off')
    
    # Noisy images from set 1
    for i in range(5):
        plt.subplot(3, 5, i + 11)
        plt.imshow(noisy_images_set1[i, 0].numpy(), cmap='gray')
        plt.title(f"Noisy First - {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('frechet_distance_example.png')
    plt.close()
    
    print(f"Example visualization saved to 'frechet_distance_example.png'")
    
    # Try different noise levels and plot the results
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    fd_values = []
    
    for noise in noise_levels:
        noisy_set = images_set1 + noise * torch.randn_like(images_set1)
        noisy_set = torch.clamp(noisy_set, 0, 1)
        
        fd_val = frechet_distance(
            images_set1, 
            noisy_set, 
            model_path='pretrained_models/svglp_smmnist2.pth',
            device=device
        )
        
        fd_values.append(fd_val)
        print(f"Noise level {noise}: Fréchet distance = {fd_val}")
    
    # Plot noise level vs Fréchet distance
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, fd_values, marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('Fréchet Distance')
    plt.title('Fréchet Distance vs. Noise Level')
    plt.grid(True)
    plt.savefig('frechet_distance_vs_noise.png')
    plt.close()
    
    print(f"Noise level analysis saved to 'frechet_distance_vs_noise.png'")
    
    # Additional experiment: Compare frames from different time steps
    print("\nComparing frames from different time steps:")
    time_diffs = []
    fd_values_time = []
    
    reference_frame_idx = 0  # Use the first frame as reference
    for frame_idx in range(0, seq_len, 2):  # Sample every other frame
        if frame_idx == reference_frame_idx:
            continue
            
        time_diff = abs(frame_idx - reference_frame_idx)
        time_diffs.append(time_diff)
        
        comparison_frames = sequences[:, frame_idx].permute(0, 3, 1, 2)
        
        fd_val = frechet_distance(
            images_set1,  # First frames
            comparison_frames,
            model_path='pretrained_models/svglp_smmnist2.pth',
            device=device
        )
        
        fd_values_time.append(fd_val)
        print(f"Time difference {time_diff}: Fréchet distance = {fd_val}")
    
    # Plot time difference vs Fréchet distance
    plt.figure(figsize=(10, 6))
    plt.plot(time_diffs, fd_values_time, marker='o')
    plt.xlabel('Time Difference (frames)')
    plt.ylabel('Fréchet Distance')
    plt.title('Fréchet Distance vs. Time Difference')
    plt.grid(True)
    plt.savefig('frechet_distance_vs_time.png')
    plt.close()
    
    print(f"Time difference analysis saved to 'frechet_distance_vs_time.png'")

if __name__ == "__main__":
    main() 