#   --- PatchGAN Discriminator ---
# Based on https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from config import IMG_WIDTH, IMG_HEIGHT
#from keras.utils.vis_utils import plot_model

def normalize_image(image):
    """0-255 aralığındaki görüntüyü -1 ile 1 arasına ölçekle"""
    return (image / 127.5) - 1.0
# define and compile the discriminator model
def define_discriminator():
    # weight initialization


    init = RandomNormal(stddev=0.02)
    
    in_src_image = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))
    in_target_image = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 2))

    merged = Concatenate()([in_src_image, in_target_image])

    # C64 (BatchNormalization kullanılmadı)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)  # Normalization burada başlıyor
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256 (512 yerine 256 kullandım)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)  # Ekstra BatchNormalization kaldırıldı

    # PatchGAN output
    d = Conv2D(1, (3, 3), padding='same', kernel_initializer=init)(d)  # 4x4 yerine 3x3 kullanıldı
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out, name='model_disc')
    
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    return model


'''
# create the model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
#plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)
'''
'''
model = define_discriminator()
model.summary()

'''
