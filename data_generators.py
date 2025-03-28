import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from skimage.color import rgb2lab
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, TEST_DIR, TRAIN_DIR, VAL_DIR
import pandas as pd
import os
import cv2
import numpy as np

TRAIN_GRAYSCALE_DIR = "./Dataset-Places2/train"
TRAIN_COLOR_DIR = "./Dataset-Places2/train"
VAL_GRAYSCALE_DIR = "./Dataset-Places2/val"
TEST_GRAYSCALE_DIR = "./Dataset-Places2/test"

IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 16

# Klasördeki grayscale ve color resimlerini al
def get_image_paths():
    grayscale_dir = "./Dataset-Places2/augmented2/grayscale"
    color_dir = "./Dataset-Places2/augmented2/color"

    grayscale_images = sorted(os.listdir(grayscale_dir))
    color_images = sorted(os.listdir(color_dir))

    df = pd.DataFrame({
        'grayscale': [os.path.join(grayscale_dir, img) for img in grayscale_images],
        'color': [os.path.join(color_dir, img) for img in color_images]
    })
    
    return df

def get_grayscale_image_paths(directory):
    grayscale_dir = os.path.join(directory, "grayscale")  # Siyah-beyaz resimlerin bulunduğu klasör
    grayscale_images = sorted(os.listdir(grayscale_dir))  # Dosyaları sıralıyoruz

    # DataFrame oluşturuyoruz, sadece grayscale resimlerin yollarını alıyoruz
    df = pd.DataFrame({
        'grayscale': [os.path.join(grayscale_dir, img) for img in grayscale_images],
    })
    
    return df


# Train, Validation ve Test için DataFrame oluştur
df_train = get_image_paths()
df_val = get_grayscale_image_paths(VAL_GRAYSCALE_DIR)
df_test = get_grayscale_image_paths(TEST_GRAYSCALE_DIR)

def custom_generator(df, batch_size, img_size, shuffle=True, seed=None):
    while True:
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Veriyi karıştır (seed ile)

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            x_batch = []
            y_batch = []
            
            for _, row in batch_df.iterrows():
                # Grayscale resmi yükle (L kanalı)
                gray_img = cv2.imread(row['grayscale'], cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    print(f"Warning: Could not read {row['grayscale']}")
                    continue
   
    
                gray_img = cv2.resize(gray_img, img_size) / 127.5 - 1.0  # Normalize [-1, 1]
                # x_batch.append(np.expand_dims(gray_img, axis=-1))  # (H, W, 1)
                x_batch.append(np.expand_dims(gray_img, axis=-1))  # (H, W, 1)

                if 'color' in df.columns:
                    color_img = cv2.imread(row.get('color', ''), cv2.IMREAD_COLOR)
                    if color_img is None:
                        print(f"Warning: Could not read {row['color']}")
                        continue #Bozuk veya eksik dosya varsa atlanır
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) 
                    color_img = cv2.resize(color_img, img_size) / 127.5 - 1.0  # Normalize [-1,1]
                    lab_img = rgb2lab(color_img) # RGB -> LAB dönüşümü 
                    ab_channels = lab_img[:, :, 1:] / 128.0  # AB kanallarını normalize et (range: [-1, 1])
                    y_batch.append(ab_channels)  # (H, W, 2) - AB kanalları
   
            if len(x_batch)==0 or len(y_batch)==0:
                continue # Eğer batch boşsa, bir sonraki iterasyona geç
            # Validation ve test için sadece x_batch döndür
            if 'color' in df.columns:
                yield np.array(x_batch),np.array(y_batch)
            else:
                yield np.array(x_batch) # y_batch olmadan döndür    
           

# Eğitim veri jeneratörünü oluştur
def create_train_generator(seed=None):
     return custom_generator(df_train, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), shuffle=True, seed=seed), len(df_train)

# Validation veri jeneratörünü oluştur
def create_val_generator(seed=None):
    return custom_generator(df_val, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), shuffle=False, seed=seed), len(df_val)

# Test veri jeneratörünü oluştur
def create_test_generator(seed=None):
    return custom_generator(df_test, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), shuffle=False, seed=seed), len(df_test)
