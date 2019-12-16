from dataset import load_dataset
from model import DiffuseModel, SpecularModel
import numpy as np
import tensorflow as tf
import argparse


# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size [default: 5]')
parser.add_argument('--patch_size', type=int, default=65, help='Patch size [default: 100]')
parser.add_argument('--kernel_size', type=int, default=21, help='Kernel size [default: 21]')
parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate [default: e-5]')
parser.add_argument('--epsilon', type=int, default=0.00316, help='Epsilon [default: 0.00316]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--out_dir', type=str, default='output', help='Output dir [default: output]')
parser.add_argument('--dataset', type=str, default='images', help='Dataset dir  [default: images]')
parser.add_argument('--model', type=str, default='KPCN', help='KPCN or DPCN [default: KPCN]')
parser.add_argument('--mode', type=str, default='train', help='test or train [default: train]')

FLAGS = parser.parse_args()

def apply_kernel(image, weights):
	pass

def train():
	# load dataset iterator
	dataset = load_dataset(FLAGS.dataset, FLAGS.batch_size)

	# initialize diffuse and specular models
	diffuse_model = DiffuseModel()
	specular_model = SpecularModel()
	diff_opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)
	spec_opt = tf.keras.optimizers.Adam(FLAGS.learning_rate)

	# for saving/loading model
	checkpoint_dir = FLAGS.log_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(diffuse=diffuse_model, specular=specular_model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # ensure the output directory exists
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
	
	for epoch in range(FLAGS.num_epoch):
		print('========================== EPOCH %d  ==========================' % epoch)
		for iteration, batch in dataset:
			with tf.GradientTape() as diff_tape, tf.GradientTape() as spec_tape:
				diff_out = diffuse_model(batch['diff'])
				spec_out = specular_model(batch['spec'])
				if FLAGS.model == 'KPCN':
					diff_out = apply_kernel(diff_out, batch['diff'])
					spec_out = apply_kernel(diff_out, batch['spec'])
				diff_loss = diffuse_model.loss(diff_out, batch['diff_gt'])
				spec_loss = specular_model.loss(spec_out, batch['spec_gt'])
				if iteration % 100:
					print('Loss: %f, %f' % (diff_loss, spec_loss))
			diff_opt.apply_gradients(zip(diff_tape.gradient(diff_loss, diffuse_model.trainable_variables), diffuse_model.trainable_variables))
			spec_opt.apply_gradients(zip(spec_tape.gradient(spec_loss, specular_model.trainable_variables), specular_model.trainable_variables))
		# save after every epoch
		manager.save()

def test():
	# load dataset iterator
	dataset = load_dataset(FLAGS.dataset, FLAGS.batch_size, is_testing=True)

	# initialize diffuse and specular models
	diffuse_model = DiffuseModel()
	specular_model = SpecularModel()

	# restore last checkpoint
	checkpoint = tf.train.Checkpoint(diffuse=diffuse_model, specular=specular_model)
	status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.log_dir))



	return

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
	if FLAGS.mode == 'train':
		train()
	if FLAGs.mode == 'test':
		test()
