import tensorflow as tf
import numpy as np

class Mind_model:

	def __init__(self, options):

		'''
		options
		n_source_quant : quantization channels of source text
		n_target_quant : quantization channels of target text
		residual_channels : number of channels in internal blocks
		batch_size : Batch Size
		sample_size : Text Sample Length
		encoder_filter_width : Encoder Filter Width
		decoder_filter_width : Decoder Filter Width
		encoder_dilations : Dilation Factor for decoder layers (list)
		decoder_dilations : Dilation Factor for decoder layers (list)
		'''

		self.options = options

		source_initializer = tf.truncated_normal_initializer(stddev=0.02)
		target_initializer = tf.truncated_normal_initializer(stddev=0.02)
		source_embedding_shape = [options['n_source_quant'], 2 * options['residual_channels']]
		target_embedding_shape = [options['n_target_quant'], options['residual_channels']]
		
		self.w_target_embedding = tf.get_variable('w_target_embedding', target_embedding_shape, initializer=target_initializer)
		self.w_source_embedding = tf.get_variable('w_source_embedding', source_embedding_shape, initializer=source_initializer)

	def encode_layer(self, input_, dilation, layer_no, last_layer=False):
	"""Utility function for forming an encode layer"""

	def encoder(self, input_):
	"""Utility function for constructing the encoder"""

	def decode_layer(self, input_, dilation, layer_no):
	"""Utility function for forming a decode layer"""

	def decoder(self, input_, encoder_embedding=None):
	"""Utility function for constructing the decoder"""

	def loss(self, decoder_output, target_sentence):
	"""Calculate loss between decoder output and target"""

	def conv1d(input_, output_channels, filter_width=1, stride=1, stddev=0.02, name='conv1d'):
	"""Helper function to create and store weights and biases with convolutional layer"""

	with tf.variable_scope(name):

		input_shape = input_.get_shape()
		input_channels = input_shape[-1]
		shape = [filter_width, input_channels, output_channels]
		weight_init = tf.truncated_normal_initializer(stddev=stddev)
		bias_init = tf.constant_initializer(0.0)

		filter_ = tf.get_variable('w', shape, initializer=weight_init)
		conv = tf.nn.conv1d(input_, filter_, stride = stride, padding = 'SAME')
		biases = tf.get_variable('biases', [output_channels], initializer=bias_init)
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

