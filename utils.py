from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from config import TEST_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, GENERATOR_DIR, DISCRIMINATOR_DIR, WEIGHTS_DIR, PLOTS_DIR, GENERATOR_PRE_DIR, DISCRIMINATOR_PRE_DIR
from data_generators import create_train_generator, create_val_generator
from skimage.color import lab2rgb


# Images from generators
def summarize_performance(step, g_model, d_model, train=True, n_samples=BATCH_SIZE, pretrained=False):
	# select a sample of input images
	# from train generator
	if train:
		train_data_gen, num_train_samples = create_train_generator()
		X_real, Y_real = next(train_data_gen)
	# from validation generator
	else:
		val_data_gen, num_val_samples = create_val_generator()
		X_real, Y_real = next(val_data_gen)

	if pretrained:
		G_X_real = tf.concat([tf.expand_dims(X_real, axis=-1)] * 3, axis=-1)
		# generate a batch of fake samples
		Y_fake = g_model.predict(G_X_real)
	else:
		# generate a batch of fake samples
		Y_fake = g_model.predict(X_real)
	Y_fake = Y_fake * 128.  # un-normalize
	#X_real = (X_real + 1.) * 50.  # un-normalize
	X_real = X_real * 100
	Y_real = Y_real * 128.  # un-normalize

	# plot real source images, grayscale
	for i in range(n_samples):

		input_img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
		input_img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
		input_img = lab2rgb(input_img)
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(input_img)

	# plot generated/predicted target image
	for i in range(n_samples):

		img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
		img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
		img[:, :, 1:] = Y_fake[i]  # predicted two layers
		try:
			img = lab2rgb(img)
		except Exception as e:
			print("Hata oluştu:", e)
			print("LAB Değerleri (min, max):", np.min(img), np.max(img))


		

		#print(X_real_train[i][:, :, 0])
		#print(Y_fake_train[i])

		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(img)

	# plot real/original target image
	for i in range(n_samples):

		img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
		img[:, :, 0] = X_real[i][:, :, 0]  # grayscale
		img[:, :, 1:] = Y_real[i]  # original two layers
		img = lab2rgb(img)

		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(img)

	# save plot to file
	if train:
		plot_filename = PLOTS_DIR + 'plot_train_%06d.png' % (step + 1)
	else:
		plot_filename = PLOTS_DIR + 'plot_val_%06d.png' % (step + 1)
	pyplot.savefig(plot_filename)
	pyplot.close()

	if pretrained:
		gen_dir = GENERATOR_PRE_DIR
		disc_dir = DISCRIMINATOR_PRE_DIR
	else:
		gen_dir = GENERATOR_DIR
		disc_dir = DISCRIMINATOR_DIR

	if train:
		# save the generator model
		gen_filename = gen_dir + 'model_pix2pix_%06d.h5' % (step + 1)
		g_model.save(gen_filename)

		# save the discriminator model
		disc_filename = disc_dir + 'model_pix2pixD_%06d.h5' % (step + 1)
		d_model.save(disc_filename)

		# Save generator weights
		g_model.save_weights(WEIGHTS_DIR + 'model_pix2pix_weights_%06d.weights.h5' % (step + 1))

		print('>Saved: %s and %s' % (plot_filename, gen_filename))