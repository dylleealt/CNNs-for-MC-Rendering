import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DiffuseModel(tf.keras.Model):
    def __init__(self, FLAGS):
        super(DiffuseModel, self).__init__()

        self.num_conv_layers = 9
        self.patch_size = FLAGS.patch_size
        self.output_channel = FLAGS.kernel_size**2 if FLAGS.model == 'KPCN' else 3
        self.filters = [100, 100, 100, 100, 100, 100, 100, 100, self.output_channels]
        self.kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5, 5]

        assert len(self.filters) == self.num_conv_layers
        assert len(self.kernel_sizes) == self.num_conv_layers

        self.weights = []        

        for i in range(self.num_conv_layers):
            activation = 'relu' if i != self.num_conv_layers - 1 else None
            self.weights.append(tf.keras.layers.Conv2D(self.filters[i], self.kernel_sizes[i], padding='same', activation=activation))

    def call(self, image):
        '''
        '''
        for conv in self.weights:
            image = conv(image)

        return image

    def loss(self, denoised, gt):

        """
        Computes the loss for a batch of images.
        :param denoised: a 4d matrix (batchsize, width, height, 3)
        :param gt: a 4d matrix (batchsize, width, height, 3)
        :return: loss, a TensorFlow scalar
        """
        return tf.reduce_mean(tf.abs(denoised - gt))


    def accuracy(self, denoised, gt):
        """
        Computes the accuracy for a batch of images.
        :param denoised: a 4d matrix (batchsize, width, height, 3)
        :param original: a 4d matrix (batchsize, width, height, 3)
        :return: accuracy, a TensorFlow scalar
        """        
        return 1.0 - (tf.reduce_mean(tf.abs(denoised - gt))

class SpecularModel(tf.keras.Model):
    def __init__(self, FLAGS):
        super(SpecularModel, self).__init__()

        self.num_conv_layers = 9
        self.patch_size = FLAGS.patch_size
        self.output_channel = FLAGS.kernel_size**2 if FLAGS.model == 'KPCN' else 3
        self.filters = [100, 100, 100, 100, 100, 100, 100, 100, self.output_channels]
        self.kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5, 5]

        assert len(self.filters) == self.num_conv_layers
        assert len(self.kernel_sizes) == self.num_conv_layers

        self.weights = []        

        for i in range(self.num_conv_layers):
            activation = 'relu' if i != self.num_conv_layers - 1 else None
            self.weights.append(tf.keras.layers.Conv2D(self.filters[i], self.kernel_sizes[i], padding='same', activation=activation))

    def call(self, image):
        '''
        '''
        for conv in self.weights:
            image = conv(image)

        return image

    def loss(self, denoised, gt):

        """
        Computes the loss for a batch of images.
        :param denoised: a 4d matrix (batchsize, width, height, 3)
        :param gt: a 4d matrix (batchsize, width, height, 3)
        :return: loss, a TensorFlow scalar
        """
        return tf.reduce_mean(tf.abs(denoised - gt))


    def accuracy(self, denoised, gt):
        """
        Computes the accuracy for a batch of images.
        :param denoised: a 4d matrix (batchsize, width, height, 3)
        :param original: a 4d matrix (batchsize, width, height, 3)
        :return: accuracy, a TensorFlow scalar
        """        
        return 1.0 - (tf.reduce_mean(tf.abs(denoised - gt))
