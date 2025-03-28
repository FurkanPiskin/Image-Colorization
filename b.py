# Gerekli kütüphaneleri import et
import tensorflow as tf
from tensorflow import keras

# Modeli yükle
g_model = keras.models.load_model(latest_gen)  # Generator modelini yükle
d_model = keras.models.load_model(latest_disc)  # Discriminator modelini yükle

# Yüklenen modelin özetini yazdır
print("Generator Model Summary:")
g_model.summary()

print("Discriminator Model Summary:")
d_model.summary()
