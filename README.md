# Stochastic Video Generation with a Learned Prior
This is code for the paper [Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus. See the [project page](https://sites.google.com/view/svglp/) for details and generated video sequences.

##  Training on Stochastic Moving MNIST (SM-MNIST)
To train the SVG-LP model on the 2 digit SM-MNIST dataset run: 
```
python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```
If the MNIST dataset doesn't exist, it will be downloaded to the specified path.

## BAIR robot push dataset
To download the BAIR robot push dataset run:
```
sh data/download_bair.sh /path/to/data/
```
This will download the dataset in tfrecord format into the specified directory. To train the pytorch models, we need to first convert the tfrecord data into .png images by running:
```
python data/convert_bair.py --data_dir /path/to/data/
```
This may take some time. Images will be saved in ```/path/to/data/processeddata```.
Now we can train the SVG-LP model by running:
```
python train_svg_lp.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```

To generate images with a pretrained SVG-LP model run:
```
python generate_svg_lp.py --model_path pretrained_models/svglp_bair.pth --log_dir /generated/images/will/save/here/
```


## KTH action dataset
First download the KTH action recognition dataset by running:
```
sh data/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want [ffmpeg](https://ffmpeg.org/) installed. The following script will do the conversion, but beware, it's written in lua (sorry!):
```
th data/convert_kth.lua --dataRoot /my/kth/data/path/ --imageSize 64
```
The ```--imageSize``` flag specifiec the image resolution. Experimental results in the paper used 128x128, but you can also train a model on 64x64 and it will train much faster.
To train the SVG-FP model on 64x64 KTH videos run:
```
python train_svg_fp.py --dataset kth --image_width  64 --model vgg --g_dim 128 --z_dim 24 --beta 0.000001 --n_past 10 --n_future 10 --channels 1 --lr 0.0008 --data_root /path/to/data/ --log_dir /logs/will/be/saved/here/
```

## Fréchet Distance Calculation
This repository includes functionality to calculate the Fréchet distance between two sets of images using the encoder part of the pretrained MMNIST model. The Fréchet distance is a measure of similarity between two distributions and is commonly used to evaluate the quality of generated images.

### Usage
You can calculate the Fréchet distance between two sets of images by importing the `frechet_distance` function from `frechet_distance.py`:

```python
import torch
from frechet_distance import frechet_distance

# Load your image sets (as torch.Tensor with shape [batch_size, channels, height, width])
# Images can be in range [0, 1] or [0, 255]
images1 = torch.load('path/to/first/image/set.pt')
images2 = torch.load('path/to/second/image/set.pt')

# Calculate Fréchet distance
distance = frechet_distance(
    images1, 
    images2, 
    model_path='pretrained_models/svglp_smmnist2.pth',  # Path to pretrained model
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Device to run on
)

print(f"Fréchet distance: {distance}")
```

The function automatically handles:
- Loading the pretrained MMNIST model
- Preprocessing the images (resizing, normalizing, etc.)
- Encoding the images using the model's encoder
- Calculating the statistics (mean and covariance) of the encoded features
- Computing the Fréchet distance between the two distributions

### Model Caching

The implementation includes a caching mechanism that stores loaded models in memory. This means that if you call the `frechet_distance` function multiple times with the same model path and device, the model will only be loaded once, which significantly improves performance for repeated calculations.

If you need to free up memory, you can clear the model cache:

```python
from frechet_distance import clear_model_cache

# Calculate distances with multiple calls (model is loaded only once)
distance1 = frechet_distance(images1, images2)
distance2 = frechet_distance(images3, images4)  # Uses cached model

# Clear the cache when done to free memory
clear_model_cache()
```

### Examples

#### Basic Example
A simple example is included at the end of `frechet_distance.py` and can be run directly:

```
python frechet_distance.py
```

This will generate two random sets of images and calculate the Fréchet distance between them.

#### Moving MNIST Example
A more comprehensive example using the Moving MNIST dataset is provided in `example_frechet_distance.py`:

```
python example_frechet_distance.py
```

This script:
1. Loads the Moving MNIST dataset (the same dataset used for training the model)
2. Extracts the first and last frames from each sequence
3. Calculates the Fréchet distance between these two sets of frames
4. Creates a noisy version of the first frames and calculates the Fréchet distance between the original and noisy frames
5. Visualizes examples from each set and saves them to `frechet_distance_example.png`
6. Tests different noise levels and plots the relationship between noise level and Fréchet distance, saving the results to `frechet_distance_vs_noise.png`
7. Compares frames from different time steps in the sequences and plots the relationship between time difference and Fréchet distance, saving the results to `frechet_distance_vs_time.png`

This example demonstrates how the Fréchet distance increases as the distributions become more dissimilar, both when adding noise and when comparing frames that are temporally further apart in the sequences.
