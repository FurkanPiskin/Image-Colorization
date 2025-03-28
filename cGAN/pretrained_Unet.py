# Based on https://www.tensorflow.org/tutorials/images/segmentation
import tensorflow as tf
import numpy as np
from config import IMG_WIDTH, IMG_HEIGHT
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
#from PatchGan_Discriminator import define_discriminator


def upsample(filters, size, apply_dropout=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_WIDTH, IMG_HEIGHT, 3], include_top=False)
print(base_model.summary())

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 128x128
    'block_3_expand_relu',   # 64x64
    'block_6_expand_relu',   # 32x32
    'block_13_expand_relu',  # 16x16
    'block_16_project',      # 8x8, the bottleneck
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name='mobilenet_down')

#down_stack.trainable = False  # comment

up_stack = [
    upsample(512, 3),                       # 8x8  ->  16x16
    upsample(256, 3),                       # 16x16 -> 32x32
    upsample(128, 3, apply_dropout=False),  # 32x32 -> 64x64
    upsample(64, 3, apply_dropout=False),   # 64x64 -> 128x128
]


def pretrained_unet_model(output_channels=2):
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 2])

    # Downsampling through the model
    skips = down_stack(inputs, training=True)  # for dropout
    x = skips[-1]
    skips = reversed(skips[:-1])  # outputs to be concatenated

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x, training=True)  # for batchnorm and dropout
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')   # 128x128 -> 256x256

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='pretrained_unet')


# define the combined generator and discriminator model, for updating the generator
def define_gan_pretrained(g_model, d_model, image_shape=(IMG_WIDTH, IMG_HEIGHT, 1)):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    # define the source image
    in_src = Input(shape=image_shape)
    #print(in_src.shape)

    # stack 3 grayscale images for generator input
    #g_input = tf.stack((tf.reshape(in_src, (-1, 256, 256)), tf.reshape(in_src, (-1, 256, 256)), tf.reshape(in_src, (-1, 256, 256))), axis=3)
    g_input =in_src
    #print(g_input.shape)
    # connect the source image to the generator input
    gen_out = g_model(g_input)

    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out], name='full_gan_pretrained')
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model

'''
d_model = define_discriminator()
g_model = pretrained_unet_model()
print(g_model.summary())
gan_model_pretrained = define_gan_pretrained(g_model, d_model)
print(gan_model_pretrained.summary())
'''

