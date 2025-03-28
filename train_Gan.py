import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import BATCH_SIZE, EPOCHS, PATCH_SHAPE, GENERATOR_DIR, DISCRIMINATOR_DIR
from data_generators import create_train_generator, create_val_generator
from cGAN.PatchGan_Discriminator import define_discriminator
from cGAN.U_net_Generator import define_generator
from cGAN.full_cGAN import define_gan
from utils import summarize_performance
import tensorflow.keras.backend as K


os.environ['TF_MIN_LOG_LEVEL'] = '2'  # minimize some log messages
tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.mixed_precision.set_global_policy('mixed_float16')        

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
    list_of_gens = glob.glob(GENERATOR_DIR + '*')
    if len(list_of_gens) > 0:
        latest_gen = max(list_of_gens, key=os.path.getctime)
        print(latest_gen)
        g_model = keras.models.load_model(latest_gen)
    else:
        g_model = define_generator()

    # if saved, load latest discriminator
    list_of_disc = glob.glob(DISCRIMINATOR_DIR + '*')
    if len(list_of_disc) > 0:
        latest_disc = max(list_of_disc, key=os.path.getctime)
        print(latest_disc)
        d_model = keras.models.load_model(latest_disc)
    else:
        d_model = define_discriminator()

    # define the composite model
    gan_model = define_gan(g_model, d_model)
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
        print(f"X_real shape: {X_real.shape}, Y_real shape: {Y_real.shape}")
        

        y_real_ones = np.ones((BATCH_SIZE, PATCH_SHAPE, PATCH_SHAPE, 1))

        # generate a batch of fake samples
        Y_fake = g_model.predict(X_real)
        y_fake_zeros = np.zeros((BATCH_SIZE, PATCH_SHAPE, PATCH_SHAPE, 1))
        print("X_real Min/Max:", X_real.min(), X_real.max())

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
            summarize_performance(i, g_model, d_model, train=True)
            summarize_performance(i, g_model, d_model, train=False)
            K.clear_session()
