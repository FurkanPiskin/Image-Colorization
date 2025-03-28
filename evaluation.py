from data_generators import create_test_generator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
import numpy as np
from config import GENERATOR_DIR, GENERATOR_PRE_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

#Bu dosya,eğitilmiş bir modeli alır,test verisi üzreinde çalıştırır,tahmin edilen (üretilmiş) görüntüleri kaydeder ve kalite ölçümlerini hesaplar
pretrained = False
GEN_PATH = GENERATOR_DIR + 'model_pix2pix_005475.h5'
#GEN_PATH = GENERATOR_PRE_DIR + 'model_pix2pix_021380.h5'
test_data_gen, num_test_samples = create_test_generator()

if pretrained:
    # Load model
    g_model = keras.models.load_model(GEN_PATH)
else:
    # Load model
    g_model = keras.models.load_model(GEN_PATH)

step = 0
psnrList = []
ssimList = []
while True:
    # Next batch
    X_real, Y_real = next(test_data_gen)
    step += 1

    if pretrained:
        #G_X_real = tf.stack((tf.reshape(X_real, (-1, 256, 256)), tf.reshape(X_real, (-1, 256, 256)),
        #                    tf.reshape(X_real, (-1, 256, 256))), axis=3)
        G_X_real = tf.concat([tf.expand_dims(X_real, axis=-1)] * 3, axis=-1)
        # generate a batch of fake samples
        Y_fake = g_model.predict(G_X_real)
    else:
        # generate a batch of fake samples
        Y_fake = g_model.predict(X_real)

    # Un-normalize
    Y_fake = Y_fake * 128.
    X_real = (X_real + 1.) * 50.
    Y_real = Y_real * 128.

    for i in range(BATCH_SIZE):
        # Original RBG image
        original_img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
        original_img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
        original_img[:, :, 1:] = Y_real[i]  # original two layers
        original_img = lab2rgb(original_img)

        # Predicted RGB image
        img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
        img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
        img[:, :, 1:] = Y_fake[i]  # predicted two layers
        img = lab2rgb(img)

        # Grayscale image
        gray_img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
        gray_img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
        gray_img = lab2rgb(gray_img)

        # Doesn't save every image
        plt.imsave('./Colorized-images/Real_evaluation_pix2pix/' + 'img' + str(step + i) + '.png', original_img)
        plt.imsave('./Colorized-images/Predicted_evaluation_pix2pix/' + 'img' + str(step + i) + 'pred.png', img)
        plt.imsave('./Colorized-images/Grayscale/' + 'img' + str(step + i) + '.png', gray_img)

        # Calculate PSNR in sRGB
        psnr = tf.image.psnr(original_img, img, max_val=1.0)
        psnrList.append(psnr.numpy())
        print("PSNR: ", psnr.numpy())

        # Calculate SSIM in sRGB
        ssim = tf.image.ssim(original_img, img, max_val=1.0)
        ssimList.append(ssim.numpy())
        print("SSIM: ", ssim.numpy())

    if step == 500:  # 2000 images, 4*500
        break


averagePSNR = sum(psnrList) / len(psnrList)
print('Average PSNR: ', averagePSNR)
print(len(psnrList))

averageSSIM = sum(ssimList) / len(ssimList)
print('Average SSIM: ', averageSSIM)