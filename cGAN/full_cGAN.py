#   --- Full cGAN Pix2Pix ---
# Based on https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

# example of defining a composite model for training the generator model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from config import IMG_WIDTH, IMG_HEIGHT
from cGAN.U_net_Generator import define_generator
from cGAN.PatchGan_Discriminator import define_discriminator



# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape=(IMG_WIDTH, IMG_HEIGHT, 1)):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = True
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out], name='full_gan')
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
	return model

'''
# define image shape
image_shape = (256,256,1)
# define the models
d_model = define_discriminator()
g_model = define_generator()
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()
'''
