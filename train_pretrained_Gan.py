import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import BATCH_SIZE, EPOCHS, PATCH_SHAPE, GENERATOR_PRE_DIR, DISCRIMINATOR_PRE_DIR
from data_generators import create_train_generator, create_val_generator
from cGAN.PatchGan_Discriminator import define_discriminator
from cGAN.pretrained_Unet import pretrained_unet_model
from cGAN.pretrained_Unet import define_gan_pretrained
from utils import summarize_performance


from tensorflow.keras.layers import Layer


class StackImages(Layer):
    def __init__(self):
        super(StackImages, self).__init__()

    def call(self, inputs):
        # inputs, X_real olacak
        # Reshape işlemi ve üç kanalın birleştirilmesi
        G_X_real = tf.concat([inputs] * 3, axis=-1)
        return G_X_real
    
os.environ['TF_MIN_LOG_LEVEL'] = '2'  # minimize some log messages

if __name__ == '__main__':

    print("Tensorflow version:", tf.__version__)

    # Check if GPU available
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Create train and validation data generators
    train_data_gen, num_train_samples = create_train_generator()
    val_data_gen, num_val_samples = create_val_generator()

    # Define, compile and fit model

    # if saved, load latest generator
    list_of_gens = glob.glob(GENERATOR_PRE_DIR + '*')
    if len(list_of_gens) > 0:
        latest_gen = max(list_of_gens, key=os.path.getctime)
        print(latest_gen)
        g_model = keras.models.load_model(latest_gen)
        g_model.trainable = True # make layers of mobilenet trainable
    else:
        g_model = pretrained_unet_model()

    # if saved, load latest discriminator
    list_of_disc = glob.glob(DISCRIMINATOR_PRE_DIR + '*')
    if len(list_of_disc) > 0:
        latest_disc = max(list_of_disc, key=os.path.getctime)
        print(latest_disc)
        d_model = keras.models.load_model(latest_disc)
    else:
        d_model = define_discriminator()

    # define the composite model
    gan_model = define_gan_pretrained(g_model, d_model)
    # summarize the model
    #gan_model.summary()

    # calculate the number of steps per training epoch
    steps_per_epo = int(num_train_samples / BATCH_SIZE)
    print('Steps per epoch: ', steps_per_epo)
    # calculate the number of training iterations, total steps
    n_steps = steps_per_epo * EPOCHS
    print('Total number of steps: ', n_steps)

    # manually enumerate epochs
    for i in range(n_steps):

        # select a batch of real samples
        X_real, Y_real = next(train_data_gen)
        y_real_ones = np.ones((BATCH_SIZE, PATCH_SHAPE, PATCH_SHAPE, 3))

        # Stack three grayscale images for generator input
        #G_X_real = tf.stack((tf.reshape(X_real, (-1, 256, 256)), tf.reshape(X_real, (-1, 256, 256)), tf.reshape(X_real, (-1, 256, 256))), axis=3)
        stack_images_layer = StackImages()
        # Train loop içinde X_real ile kullanın
        G_X_real = stack_images_layer(X_real)

        # generate a batch of fake samples
        Y_fake = g_model.predict(G_X_real)
        y_fake_zeros = np.zeros((BATCH_SIZE, PATCH_SHAPE, PATCH_SHAPE, 3))

        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_real, Y_real], y_real_ones)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_real, Y_fake], y_fake_zeros)
        # update the generator
        # for output we want all ones from the disc. and the real color img (X, Y)
        g_loss, _, _ = gan_model.train_on_batch(X_real, [y_real_ones, Y_real])
        # summarize performance
        print('Training step %d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        # New epoch starting
        if (i+1) % steps_per_epo == 0:
            # Start generator again for new epoch
            print('Init generator again...')
            train_data_gen, _ = create_train_generator()

            # Plots, save model, validation
            summarize_performance(i, g_model, d_model, train=True, pretrained=True)
            summarize_performance(i, g_model, d_model, train=False, pretrained=True)
