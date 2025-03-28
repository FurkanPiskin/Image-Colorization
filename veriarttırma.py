import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2

# Veri klasörlerini tanımla
train_color_dir = "./Dataset-Places2/train/color"  # Renkli görüntüler
train_gray_dir = "./Dataset-Places2/train/grayscale"  # Gri görüntüler
augmented_color_dir = "./Dataset-Places2/augmented2/color"  # Artırılmış renkli görüntüler
augmented_gray_dir = "./Dataset-Places2/augmented2/grayscale"  # Artırılmış gri görüntüler

# Eğer artırılmış veri klasörleri yoksa oluştur
os.makedirs(augmented_color_dir, exist_ok=True)
os.makedirs(augmented_gray_dir, exist_ok=True)

# Veri artırma için dönüşümler
data_gen = ImageDataGenerator(
    rotation_range=20,  # Rasgele döndürme (-20, +20 derece)
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    shear_range=0.2,  # Kesme dönüşümü
    zoom_range=0.2,  # Yakınlaştırma
    horizontal_flip=True,
    vertical_flip=True,  # Yatay çevirme
    fill_mode="nearest"  # Kenarlarda en yakın pikselleri kullan
)

# Tüm görüntüleri yükle ve artırma uygula
for img_name in os.listdir(train_color_dir):
    # Renkli görüntüyü oku
    color_path = os.path.join(train_color_dir, img_name)
    gray_path = os.path.join(train_gray_dir, img_name)  # Aynı isimde gri versiyon olmalı

    if not os.path.exists(gray_path):
        print(f"Hata: {gray_path} bulunamadı, atlanıyor...")
        continue

    # OpenCV ile görüntüleri oku
    color_img = cv2.imread(color_path)
    gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı olarak oku

    # Orijinal görüntüyü kaydet
    original_color_path = os.path.join(augmented_color_dir, f"orig_{img_name}")
    original_gray_path = os.path.join(augmented_gray_dir, f"orig_{img_name}")
    cv2.imwrite(original_color_path, color_img)
    cv2.imwrite(original_gray_path, gray_img)

    # OpenCV'den numpy array formatına çevir
    color_img = np.expand_dims(color_img, axis=0)  # Batch için ek boyut
    gray_img = np.expand_dims(gray_img, axis=-1)  # 1 kanal ekle
    gray_img = np.expand_dims(gray_img, axis=0)  # Batch ekle

    # Veri artırma için iterator oluştur
    it_color = data_gen.flow(color_img, batch_size=1)
    it_gray = data_gen.flow(gray_img, batch_size=1)

    # 10 farklı artırılmış veri üret
    for i in range(10):
        aug_color = it_color.__next__()[0].astype(np.uint8)  # Artırılmış renkli görüntü
        aug_gray = it_gray.__next__()[0].squeeze().astype(np.uint8)  # Artırılmış gri görüntü

        # Yeni dosya adını oluştur
        aug_color_path = os.path.join(augmented_color_dir, f"aug_{i}_{img_name}")
        aug_gray_path = os.path.join(augmented_gray_dir, f"aug_{i}_{img_name}")

        # Görüntüleri kaydet
        cv2.imwrite(aug_color_path, aug_color)
        cv2.imwrite(aug_gray_path, aug_gray)

print("Veri artırma tamamlandı!")
