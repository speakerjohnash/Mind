import tensorflow as tf
import numpy as np

class TruthModel:

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

		if 'source_mask_chars' in options:

			# For embedding only, the input sentence before the padding
			# the output network would be conditioned only upto input length
			# also loss needs to be calculated upto target sentence

			# What is this?
			input_sentence_mask = np.ones((options['n_source_quant'], 2 * options['residual_channels']), dtype = 'float32')
			input_sentence_mask[options['source_mask_chars'], :] = np.zeros((1,2 * options['residual_channels'] ), dtype = 'float32')

			# What is this?
			output_sentence_mask = np.ones( (options['n_target_quant'], 1), dtype = 'float32')
			output_sentence_mask[options['target_mask_chars'], :] = np.zeros((1,1), dtype = 'float32')

			# What is this?
			self.input_mask = tf.constant(input_sentence_mask)
			self.output_mask = tf.constant(output_sentence_mask)

	def build_prediction_model(self):
		"""Train just the decoder"""
		
		options = self.options
		batch_size = options['batch_size']
		sample_size = options['sample_size']

		shape = [batch_size, sample_size]
		sentence = tf.placeholder('int32', shape, name='sentence')

		# Is this Correct?
		slice_shape = [batch_size, sample_size - 1]
		source_sentence = tf.slice(sentence, [0, 0], slice_shape, name='source_sentence')
		target_sentence = tf.slice(sentence, [0, 1], slice_shape, name='target_sentence')

		# Lookup Character Embeddings
		source_embedding = tf.nn.embedding_lookup(
			self.w_source_embedding, 
			source_sentence, 
			name="source_embedding"
		)

		decoder_output = self.decoder(source_embedding)
		loss = self.loss(decoder_output, target_sentence)
		
		tf.summary.scalar('loss', loss)

		flat_logits = tf.reshape(decoder_output, [-1, options['n_target_quant']])
		prediction = tf.argmax(flat_logits, 1)
		
		variables = tf.trainable_variables()
		
		tensors = {
			'sentence' : sentence,
			'loss' : loss,
			'prediction' : prediction,
			'variables' : variables
		}

		return tensors

	def build_translation_model(self):
		"""Train the encoder and the decoder"""

	def build_truth_model(self):
		"""Train the encoder, the decoder, and the memory state"""

	def encode_layer(self, input_, dilation, layer_no, last_layer=False):
		"""Utility function for forming an encode layer"""

		options = self.options

		# Reduce Dimension
		relu1 = tf.nn.relu(input_, name = 'enc_relu1_layer{}'.format(layer_no))
		conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))

		# What is this?
		conv1 = tf.mul(conv1, self.source_masked_d)
		
		# Unmasked 1 x k dilated convolution
		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = ops.dilated_conv1d(relu2, options['residual_channels'], 
			dilation, options['encoder_filter_width'],
			causal = False, 
			name = "enc_dilated_conv_layer{}".format(layer_no)
		)

		# What is this?
		dilated_conv = tf.mul(dilated_conv, self.source_masked_d)

		# Restore Dimension
		relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu3_layer{}'.format(layer_no))
		conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))

		# Residual connection
		return input_ + conv2

	def decode_layer(self, input_, dilation, layer_no):
		"""Utility function for forming a decode layer"""

		options = self.options

		# Input dimension
		in_dim = input_.get_shape().as_list()[-1]

		# Reduce dimension
		relu1 = tf.nn.relu(input_, name = 'dec_relu1_layer{}'.format(layer_no))
		conv1 = conv1d(relu1, in_dim / 2, name = 'dec_conv1d_1_layer{}'.format(layer_no))

		# Masked 1 x k dilated convolution
		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = dilated_conv1d(
			relu2,
			output_channels = in_dim / 2,
			dilation        = dilation,
			filter_width    = options['decoder_filter_width'],
			causal          = True,
			name            = "dec_dilated_conv_layer{}".format(layer_no))

		# Restore dimension
		relu3 = tf.nn.relu(dilated_conv, name = 'dec_relu3_layer{}'.format(layer_no))
		conv2 = conv1d(relu3, in_dim, name = 'dec_conv1d_2_layer{}'.format(layer_no))

		# Residual connection
		return input_ + conv2

	def encoder(self, input_):
		"""Utility function for constructing the encoder"""

		options = self.options
		curr_input = input_

		# Connect encoder layers
		for layer_no, dilation in enumerate(self.options['encoder_dilations']):

			layer_output = self.encode_layer(curr_input, dilation, layer_no)

			# Encode only until the input length, conditioning should be 0 beyond that
			layer_output = tf.mul(layer_output, self.source_masked, name = 'layer_{}_output'.format(layer_no))

			curr_input = layer_output
		
		# What is this?
		processed_output = conv1d(
			tf.nn.relu(layer_output), 
			options['residual_channels'], 
			name='encoder_post_processing'
		)

		# What are these?
		processed_output = tf.nn.relu(processed_output)
		processed_output = tf.mul(processed_output, self.source_masked_d, name='encoder_processed')

		return processed_output

	def decoder(self, input_, encoder_embedding=None):
		"""Utility function for constructing the decoder"""

		options = self.options
		curr_input = input_

		# Condition with encoder embedding for truth and translation models
		if encoder_embedding != None:
			curr_input = tf.concat(2, [input_, encoder_embedding])
			print("Decoder Input", curr_input)
			
		for layer_no, dilation in enumerate(options['decoder_dilations']):
			layer_output = self.decode_layer(curr_input, dilation, layer_no)
			curr_input = layer_output

		processed_output = conv1d(
			tf.nn.relu(layer_output), 
			options['n_target_quant'], 
			name='decoder_post_processing'
		)

		# Where is Droppout?

		return processed_output

	def loss(self, decoder_output, target_sentence):
		"""Calculate loss between decoder output and target"""

		options = self.options

		target_one_hot = tf.one_hot(
			target_sentence, 
			depth=options['n_target_quant'], 
			dtype=tf.float32
		)

		flat_logits = tf.reshape(decoder_output, [-1, options['n_target_quant']])
		flat_targets = tf.reshape(target_one_hot, [-1, options['n_target_quant']])

		# Calculate Loss
		loss = tf.nn.softmax_cross_entropy_with_logits(
			logits=flat_logits, 
			labels=flat_targets, 
			name='decoder_cross_entropy_loss'
		)

		# Mask loss beyond EOL in target
		if 'target_mask_chars' in options:
			target_masked = tf.reshape(self.target_masked, [-1])
			loss = tf.mul(loss, target_masked, name='masked_loss')
			loss = tf.div(tf.reduce_sum(loss), tf.reduce_sum(target_masked), name="Reduced_mean_loss")
		else:
			loss = tf.reduce_mean(loss, name="Reduced_mean_loss")

		return loss

	def build_generator(self, sample_size, reuse=False):
		"""Build a generator to produce thoughts"""

		if reuse:
			tf.get_variable_scope().reuse_variables()

		options = self.options

		source_sentence = tf.placeholder('int32', [1, sample_size], name='sentence')
		source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence, name="source_embedding")

		decoder_output = self.decoder(source_embedding)
		flat_logits = tf.reshape(decoder_output, [-1, options['n_target_quant']])

		prediction = tf.argmax(flat_logits, 1)
		probs = tf.nn.softmax(flat_logits)

		tensors = {
			'source_sentence': source_sentence,
			'prediction': prediction,
			'probs': probs
		}

		return tensors

# Utility Functions

def conv1d(input_, output_channels, filter_width=1, stride=1, stddev=0.02, name='conv1d'):
	"""Helper function to create and store weights and biases with convolutional layer"""

	with tf.variable_scope(name):

		input_shape = input_.get_shape()
		input_channels = input_shape[-1]
		shape = [filter_width, input_channels, output_channels]
		weight_init = tf.truncated_normal_initializer(stddev=stddev)
		bias_init = tf.constant_initializer(0.0)

		filter_ = tf.get_variable('w', shape, initializer=weight_init)
		conv = tf.nn.conv1d(input_, filter_, stride=stride, padding='SAME')
		biases = tf.get_variable('biases', [output_channels], initializer=bias_init)
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def atrous_conv1d(input_, output_channels, rate=1, filter_width=1, stride=[1], stddev=0.02, name='atrous_conv1d'):
	"""Helper function to create and store weights and biases with convolutional layer"""

	with tf.variable_scope(name):

		input_shape = input_.get_shape()
		input_channels = input_shape[-1]
		shape = [filter_width, input_channels, output_channels]
		weight_init = tf.truncated_normal_initializer(stddev=stddev)
		bias_init = tf.constant_initializer(0.0)

		filter_ = tf.get_variable('w', shape, initializer=weight_init)
		conv = tf.nn.convolution(
			input=input_, 
			filter=filter_, 
			padding="VALID", 
			dilation_rate=np.broadcast_to(rate, (1, )), 
			strides=stride, 
			name=name
		)

		biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

	return conv

def dilated_conv1d(input_, output_channels, dilation, filter_width=1, causal=False, name='dilated_conv'):
	
	# Padding for masked convolutions
	if causal:
		padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
		padded = tf.pad(input_, padding)
	else:
		padding = [[0, 0], [(filter_width - 1) * dilation/2, (filter_width - 1) * dilation/2], [0, 0]]
		padded = tf.pad(input_, padding)
	
	d_conv = atrous_conv1d(padded, output_channels, filter_width=filter_width, rate=dilation, name=name)	
	result = tf.slice(d_conv,[0, 0, 0],[-1, int(input_.get_shape()[1]), -1])
	
	return result