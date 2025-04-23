import os
import glob
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab,lab2rgb
import torch
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import warnings

class ColorizationDataset(Dataset):
    
    """
    A custom Dataset class for loading and processing images for colorization tasks.

    This class takes in a list of image file paths and a crop size. It loads images, converts them to the L*a*b color space,
    and normalizes them to be used for training a colorization model.

    Attributes:
        paths (list of str): List of file paths to the images.
        size (int or tuple): Size to which images will be cropped. If int, it is assumed to be a square crop.
    """

    def __init__(self, paths, crop_size):
        """
        Initializes the dataset with image paths and crop size.

        Args:
            paths (list of str): List of file paths to the images.
            crop_size (int or tuple): The size to crop the images to. If an int, it crops to a square of size crop_size.
        """
        self.paths = paths
        self.size = crop_size

    def __getitem__(self, idx):
        """
        Retrieves and processes an image at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'L' (torch.Tensor): Normalized luminance channel (L) of the image.
                - 'ab' (torch.Tensor): Normalized color channels (a and b) of the image.
        """
        # Open and convert the image to RGB format
        img = Image.open(self.paths[idx]).convert("RGB")
        
        # Crop the image to the specified size
        img = transforms.Compose([transforms.CenterCrop(self.size)])(img)
        
        # Convert the image to a NumPy array
        img = np.array(img)
        
        # Convert the RGB image to L*a*b color space
        img_lab = rgb2lab(img).astype("float32")
        
        # Convert the image to a PyTorch tensor
        img_lab = transforms.ToTensor()(img_lab)
        
        # Normalize the L channel (luminance) to be between -1 and 1
        # Original range: [0, 100], Target range: [-1, 1]
        L = img_lab[0, ...] / 50. - 1.  # 50 is half of 100, so this scales [0, 100] to [-1, 1]
        # output:- (batch_size, H, W)
        
        # Normalize the a and b channels (color information) to be between -1 and 1
        # Original range: [-128, 127], Target range: [-1, 1]
        ab = (img_lab[1:3, ...] + 128.) / 255. * 2. - 1.  # Offset by 128, scale to [0, 1], then map to [-1, 1]
        # output:- (batch_size, channels, H, W)

        # Return the processed image as a dictionary
        return {'L': L.unsqueeze(0), 'ab': ab}

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.paths)