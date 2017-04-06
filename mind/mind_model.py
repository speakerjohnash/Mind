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

