import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb
import os
from Unet_Generator import UNetGenerator
from Visualize_Epoch import lab_to_rgb
SIZE=256

test_dir = './Test_Images/images_test_rev1'
val_paths = [os.path.join(test_dir, test_path) for test_path in os.listdir(test_dir)]

# Choose a random image path from the testing image directory
img_test_path = np.random.choice(val_paths)

# Open and convert the image to RGB format
img = Image.open(img_test_path).convert("RGB")

# Crop the image to the specified size
img = transforms.Compose([transforms.CenterCrop(SIZE)])(img)

# Convert the image to a NumPy array
img = np.array(img)

# Convert the RGB image to L*a*b color space
img_lab = rgb2lab(img).astype("float32")

# Convert the image to a PyTorch tensor
img_lab = transforms.ToTensor()(img_lab)

# Normalize the L channel (luminance) to be between -1 and 1
L = img_lab[[0], ...] / 50. - 1.

# Add the batch dimension to the L tensor
L = L.unsqueeze(0)

# Load the pre-trained model weight for generator and set it to evaluation mode
generator = UNetGenerator(input_c=1, output_c=2, n_down=8, num_filters=64)
generator.load_state_dict(torch.load('./Generator Checkpoints/generator_epoch_17.pth'))#Choose a pth file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator.eval()

# Get ab output by passing L to generator model and removes it from the computation graph to prevent further gradient calculations
ab = generator(L.to(device))
ab = ab.detach()

# Convert the L and ab channels back to RGB color space
rgb_out = lab_to_rgb(L.to(device), ab.to(device))

# Visualize side by side BW and Colorful Image
plt.figure(figsize=(10, 10))

# BW  Image
plt.subplot(121)
plt.imshow(L[0].permute(1, 2, 0), cmap='gray')
plt.title('Black and White Image')
plt.axis('off')

# Colorful Image by AI
plt.subplot(122)
plt.title('AI Generated Colorful Image')
plt.imshow(rgb_out[0])
plt.axis('off')

# Adjust the layout
plt.tight_layout()
plt.show()