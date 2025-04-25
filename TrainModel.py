from loss_utils import create_loss_meters, update_losses
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "./Galaxies/images_training_rev1/*jpeg"
paths = glob.glob(dataset_path)

TOTAL_SAMPLES = 2000
TRAIN_SAMPLES_PERCENTAGE = 0.9
N_TRAINING_SAMPLES = int(TOTAL_SAMPLES * TRAIN_SAMPLES_PERCENTAGE)

np.random.seed(123)
paths_subset = np.random.choice(paths, TOTAL_SAMPLES, replace=False)

SIZE = 256
np.random.seed(123)

rand_idxs = np.random.permutation(TOTAL_SAMPLES)
train_idxs = rand_idxs[:N_TRAINING_SAMPLES]
val_idxs = rand_idxs[N_TRAINING_SAMPLES:]

train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]

input_nc = 1
output_nc = 2
ngf = 64
netG = UNetGenerator(input_nc, output_nc, ngf)(torch.randn(16, 1, 256, 256))

discriminator = PatchDiscriminator(input_c=3, n_down=3, num_filters=64)
dummy_input = torch.randn(16, 3, 256, 256)
out = discriminator(dummy_input)

def train_model(model, train_dl, val_dl, epochs, save_every=5):  # 🔄 val_dl parametresi eklendi
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()
        model.train()
        
        for data in tqdm(train_dl, desc=f"Epoch {e+1}/{epochs}"):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))

        print(f"\nEpoch {e+1}/{epochs} Losses:")
        log_results(loss_meter_dict)

        # ✅ Train verisi ile görselleştirme
        visualize(model, data, save=True)

        # ✅ Validation verisi ile görselleştirme (sadece ilk batch)
        model.eval()
        with torch.no_grad():
            for val_data in val_dl:
                model.setup_input(val_data)
                model.forward()
                visualize(model, val_data, save=True)
                break

        if (e + 1) % save_every == 0:
            checkpoint_path = f"./Generator_Checkpoints/generator_epoch_{e + 1}.pth"
            torch.save(model.net_G.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    return loss_meter_dict

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    train_dl = make_dataloaders(paths=train_paths, crop_size=SIZE)
    val_dl = make_dataloaders(paths=val_paths, crop_size=SIZE)

    model = MainModel()
    loss_meter_dict_output = train_model(model, train_dl, val_dl, epochs=20, save_every=1)

    """
    Loss_D_real:D'nin gerçek veriyi tanıma başarısı
    loss_D_fake:D'nin sahte veriyi tanıma başarısı
    loss_D:D'nin genel başarısı
    loss_G_GAN:G'nin sahteyi gerçek gibi yapma başarısı
    loss_G_L1:G'nin gerçek renklere yakınlığı
    loss_G:G'nin toplam hatası
    """
