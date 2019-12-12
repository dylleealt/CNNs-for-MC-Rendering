from dataset import get_data
from model import Denoise

import numpy as np
import tensorflow as tf
import argparse
import cv2

# will change hyperparameters later
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size [default: 5]')
parser.add_argument('--patch_size', type=int, default=100, help='Patch size [default: 100]')
parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate [default: e-5]')
parser.add_argument('--epsilon', type=int, default=0.00316, help='Epsilon [default: 0.00316]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--dataset', type=str, default='images', help='Dataset path  [default: images]')
FLAGS = parser.parse_args()

NUM_EPOCH = FLAGS.num_epoch
BATCH_SIZE = FLAGS.batch_size
PATCH_SIZE = FLAGS.patch_size
LEARNING_RATE = FLAGS.learning_rate
EPSILON = FLAGS.epsilon
LOG_DIR = FLAGS.log_dir
DATASET = FLAGS.dataset

def train():
	'''
	general structure for training
	'''
	inputs_diff, inputs_spec, inputs_alb, labels = get_data(DATASET)
	model = Denoise()
	optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
	#saver = tf.train.Saver()
	num_images = 1
	width = inputs_diff.shape[0]
	height = inputs_diff.shape[1]
	num_channels = inputs_diff.shape[2]
	inputs_diff = tf.reshape(inputs_diff, (num_images, width, height, num_channels))
	inputs_spec = tf.reshape(inputs_spec, (num_images, width, height, num_channels))
	inputs_alb = tf.reshape(inputs_alb, (num_images, width, height, num_channels))
	labels = tf.reshape(labels, (num_images, width, height, num_channels))
    
	inputs_diff = tf.math.divide(inputs_diff, inputs_alb + EPSILON)
	inputs_spec = tf.math.log(inputs_spec + 1)
	for epoch in range(NUM_EPOCH):
		for i in range(0, num_images):
		#for i in range(0, num_images, BATCH_SIZE):
			for j in range(0, width, PATCH_SIZE):
				for k in range(0, height, PATCH_SIZE):
					with tf.GradientTape() as tape:
						#batch_patch_inputs_diff = inputs_diff[:][j:j+PATCH_SIZE][k:k+PATCH_SIZE]
						batch_patch_inputs_diff = tf.slice(inputs_diff, begin=[0, j, k, 0], size=[num_images, PATCH_SIZE, PATCH_SIZE, num_channels])
						batch_patch_inputs_spec = tf.slice(inputs_spec, begin=[0, j, k, 0], size=[num_images, PATCH_SIZE, PATCH_SIZE, num_channels])
						batch_patch_inputs_alb = tf.slice(inputs_alb, begin=[0, j, k, 0], size=[num_images, PATCH_SIZE, PATCH_SIZE, num_channels])
						batch_patch_labels = tf.slice(labels, begin=[0, j, k, 0], size=[num_images, PATCH_SIZE, PATCH_SIZE, num_channels])
						diffuse, specular = model.call(batch_patch_inputs_diff, batch_patch_inputs_spec)
						predictions = construct_image(diffuse, specular, batch_patch_inputs_alb)
						loss = model.loss(predictions, batch_patch_labels)
					gradients = tape.gradient(loss, model.trainable_variables)
					optimizer.apply_gradients(zip(gradients, model.trainable_variables))
					print("LOSS [", epoch, ",", i, ",", j, ",", k, "]:", loss)


	# dont quite remember how to save, need sessions?
	# save_path = saver.save(, os.path.join(LOG_DIR, "model.ckpt"))
	# print("Model saved in file: %s" % save_path)
	write_prediction(inputs_diff, inputs_spec, inputs_alb, model)

def write_prediction(inputs_diff, inputs_spec, inputs_alb, model):
	prediction_diff, prediction_spec = model.call(inputs_diff, inputs_spec)
	diff = np.clip(np.array(prediction_diff), 0, 1)
	spec = np.clip(np.array(prediction_spec), 0, 1)
	alb = np.clip(np.array(inputs_alb), 0, 1)
	prediction = np.array(construct_image(diff, spec, alb)) * 255
	prediction = prediction.astype(np.uint8)
	prediction = np.clip(np.reshape(prediction, (prediction.shape[1], prediction.shape[2], 3)), 0, 255)

	s = 'images/predicted_image.png'
	cv2.imwrite(s, prediction)

def construct_image(inputs_diff, inputs_spec, inputs_alb):
	return tf.math.multiply(EPSILON + inputs_alb, inputs_diff) + tf.math.exp(inputs_spec) - 1

if __name__ == '__main__':
	train()
