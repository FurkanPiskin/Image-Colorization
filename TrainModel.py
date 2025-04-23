from loss_utils import create_loss_meters,update_losses
from tqdm import tqdm
import torch
from Visualize_Epoch import visualize
from Log_Results import log_results
import warnings
import glob
import numpy as np
from DataLoader import make_dataloaders
from MainModel import MainModel
from Unet_Generator import UNetGenerator
from PatchDiscriminator import PatchDiscriminator

warnings.filterwarnings('ignore')

# If a CUDA-compatible GPU is available, use it; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

dataset_path="./Galaxies/images_training_rev1/*jpg"
paths=glob.glob(dataset_path)
#print(len(paths))

#Subset of total Images
TOTAL_SAMPLES=20000
TRAIN_SAMPLES_PERCENTAGE=0.9
N_TRAINING_SAMPLES=int(TOTAL_SAMPLES*TRAIN_SAMPLES_PERCENTAGE)

np.random.seed(123)
#RASTGELE 20.000 verinin
paths_subset=np.random.choice(paths,TOTAL_SAMPLES,replace=False)

SIZE = 256

np.random.seed(123)

# Create a shuffled list of indices from 0 to 19,999 to randomly shuffle the paths
rand_idxs = np.random.permutation(TOTAL_SAMPLES)

# Indices for Splitting Images into train and val set
train_idxs = rand_idxs[:N_TRAINING_SAMPLES] # First 18K indices are for the train set
val_idxs = rand_idxs[N_TRAINING_SAMPLES:] # Remaining 2K for the val set

# Use these indices to get the file paths for the training and validation sets
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]

input_nc=1
output_nc=2
ngf=64
netG = UNetGenerator(input_nc, output_nc, ngf)(torch.randn(16, 1, 256, 256))

# Modeli başlatıyoruz
discriminator = PatchDiscriminator(input_c=3, n_down=3, num_filters=64)

# Dummy input verisi
dummy_input = torch.randn(16, 3, 256, 256)

# Çıktıyı alıyoruz
out = discriminator(dummy_input)

def train_model(model, train_dl, epochs, save_every=5):
    """
    Train the given model over a specified number of epochs and save checkpoints periodically.

    Parameters
    ----------
    model : MainModel
        The model object to be trained.
    train_dl : DataLoader
        DataLoader object for loading the training data.
    epochs : int
        Number of epochs to train the model.
    save_every : int, optional
        Frequency (in epochs) to save model checkpoints (default is 5).

    Returns
    -------
    loss_meter_dict : dict
        Dictionary containing AverageMeter objects with loss metrics for the final epoch.
    """
    # Loop through each epoch
    for e in range(epochs):
        # Initialize loss meters for tracking loss values for the epoch
        loss_meter_dict = create_loss_meters()
        
        # Set the model to training mode (enables features like dropout and batch normalization)
        model.train()
        
        # Iterate through the training data
        for data in tqdm(train_dl, desc=f"Epoch {e+1}/{epochs}"):
            # Prepare the input data for the model
            model.setup_input(data)
            
            # Perform a single optimization step (forward pass, loss calculation, and backpropagation)
            model.optimize()
            
            # Update the loss meters with the latest losses
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
        
        # Print out the average loss metrics for the epoch
        print(f"\nEpoch {e+1}/{epochs} Losses:")
        log_results(loss_meter_dict)
        
        # Visualize the model's outputs for the last batch of this epoch
        visualize(model, data, save=True)
        
        # Save the model's state dictionary to a checkpoint file every `save_every` epochs
        if (e + 1) % save_every == 0:
            # Define the path where the model checkpoint will be saved
            checkpoint_path = f"./Generator_Checkpoints/generator_epoch_{e + 1}.pth"
            
            # Save the generator model's state dictionary to the checkpoint path
            torch.save(model.net_G.state_dict(), checkpoint_path)
            
            # Print confirmation of the checkpoint save
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Return the dictionary of loss meters for the final epoch
    return loss_meter_dict


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Windows için güvenlik önlemi

    # DataLoader'ları oluştur
    train_dl = make_dataloaders(paths=train_paths, crop_size=SIZE)
    val_dl = make_dataloaders(paths=val_paths, crop_size=SIZE)

    model = MainModel()
    loss_meter_dict_output = train_model(model, train_dl, epochs=20, save_every=1)