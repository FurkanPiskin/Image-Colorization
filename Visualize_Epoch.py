import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.color import lab2rgb
import time

def lab_to_rgb(L, ab):
    """
    Convert a batch of L*a*b images to RGB format.

    Parameters
    ----------
    L : torch.Tensor
        The luminance (L) channel of the image.
    ab : torch.Tensor
        The color channels (a and b) of the image.

    Returns
    -------
    numpy.ndarray
        An array of RGB images.
    """
    
    # Convert L channel back from normalized range [-1, 1] to [0, 100]
    L = (L + 1) * 50
    # Convert a and b channels back from normalized range [-1, 1] to [-128, 127]
    ab = (ab + 1) * 255 / 2 - 128
    # Combine L, a, and b channels into a single tensor and convert to numpy array
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)  # Convert from L*a*b to RGB using an external function
        rgb_imgs.append(img_rgb)  # Append the RGB image to the list
    return np.stack(rgb_imgs, axis=0)  # Stack the list into a single numpy array

def visualize(model, data, save=True):
    """
    Visualize a batch of images by comparing grayscale input, generated color images, and real color images.

    Parameters
    ----------
    model : nn.Module
        The model used to generate color images.
    data : dict
        A dictionary containing input data with 'L' (grayscale) and 'ab' (real color).
    save : bool
        Whether to save the visualization as an image file (default is True).
    """
    model.net_G.eval()  # Set the generator model to evaluation mode
    with torch.no_grad():
        model.setup_input(data)  # Prepare the input data for the model
        model.forward()  # Generate color images using the model
    
    model.net_G.train()  # Set the generator model back to training mode
    fake_color = model.fake_color.detach()  # Get the generated color images without tracking gradients
    real_color = model.ab  # Get the real color images
    L = model.L  # Get the grayscale images
    
    # Convert the images to RGB format for visualization
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    
    # Create a figure with subplots to display images
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)  # Subplot for grayscale image
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        
        ax = plt.subplot(3, 5, i + 1 + 5)  # Subplot for generated color image
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        
        ax = plt.subplot(3, 5, i + 1 + 10)  # Subplot for real color image
        ax.imshow(real_imgs[i])
        ax.axis("off")
    
    plt.show()  # Display the images
    
    if save:
        fig.savefig(f"./Saved_Images/colorization_{time.time()}.png")  # Save the visualization as a PNG file