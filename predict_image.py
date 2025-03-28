import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_WIDTH, IMG_HEIGHT
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model

# Paths
IMG_PATH = r"C:\Users\bozku\Desktop\Image_Colorization\Image-colorization\Places365.jpg"  # Renklendirilmek istenen gri görüntü
GEN_PATH = r"C:\Users\bozku\Desktop\Image_Colorization\Image-colorization\Saved-models\Saved-gen\model_pix2pix_001368.h5"
SAVE_PATH = r"C:\Users\bozku\Desktop\Image_Colorization\Image-colorization\output"  # Üretilen renkli görüntünün kaydedileceği klasör

# Read image
img = imread(IMG_PATH)
img = np.stack([img] * 3, axis=-1)  # Gri tonlamalı görüntüyü RGB formatına çevir
print("Görüntü şekli:", img.shape)
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)  # Görüntüyü modelin beklediği boyuta yeniden boyutlandır

# RGB görüntüyü LAB formatına çevir
img_lab = rgb2lab(img)  # LAB renk uzayına çevir
img_l = img_lab[:, :, 0]  # L (Aydınlık) bileşeni
img_ab_real = img_lab[:, :, 1:]  # Gerçek AB kanalları (karşılaştırma için)

# Modelin beklediği forma getirme (normalize et)
X_input = img_l / 50.0 - 1.0  # [-1, 1] aralığına normalize et
X_input = np.expand_dims(X_input, axis=0)  # Batch boyutunu ekle
X_input = np.expand_dims(X_input, axis=-1)  # Kanal boyutunu ekle (1 kanal)

# Modelden tahmin al
g_model = load_model(GEN_PATH)
print(g_model.summary())  # Modelin yapısını yazdırarak doğru şekilde yüklendiğini kontrol edin

Y_fake = g_model.predict(X_input)  # Modelin ürettiği AB bileşenleri
Y_fake = Y_fake * 128  # Daha farklı bir un-normalize yöntemi deneyebilirsiniz
  # Çıkışı orijinal ölçeğe getir (normalize işlemi geri al)

# Modelin tahmin ettiği LAB görüntüsünü oluştur
pred_lab = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
pred_lab[:, :, 0] = img_l  # Orijinal L bileşeni
pred_lab[:, :, 1:] = Y_fake[0]  # Modelin ürettiği AB bileşenleri
pred_rgb = lab2rgb(pred_lab)  # LAB → RGB dönüşümü

# Gerçek renkli görüntü (LAB → RGB dönüşümü)
real_lab = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
real_lab[:, :, 0] = img_l  # Aynı L kanalı
real_lab[:, :, 1:] = img_ab_real  # Gerçek AB kanalları
real_rgb = lab2rgb(real_lab)  # Gerçek RGB görüntü

# 3'lü Görselleştirme
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(img_l, cmap='gray')  # Orijinal gri ölçekli görüntü
ax[0].set_title("Gri Ölçekli Giriş")
ax[0].axis("off")

ax[1].imshow(pred_rgb)  # Modelin tahmini renkli görüntüsü
ax[1].set_title("Modelin Ürettiği")
ax[1].axis("off")

ax[2].imshow(real_rgb)  # Gerçek renkli görüntü (karşılaştırma için)
ax[2].set_title("Gerçek Renkli Görüntü")
ax[2].axis("off")

plt.show()
