import sys

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

		source_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
		target_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
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
		
	def build_truth_model(self, sample_size):
		"""Train the encoder, the decoder, and the memory state"""

		self.options["sample_size"] = sample_size

		options = self.options
		batch_size = options["batch_size"]
		sample_size = options["sample_size"]
		latent_dims = options["latent_dims"]
		residual_channels = options["residual_channels"]

		source_size = [batch_size, options["sample_size"]]
		target_size = [batch_size, options["sample_size"] + 1]
		source_sentence = tf.placeholder("int32", source_size, name="source_sentence")
		target_sentence = tf.placeholder("int32", target_size, name="target_sentence")
		kl_weight = tf.placeholder(tf.float32, shape=[], name="kl_weight")
		phase = tf.placeholder(tf.bool, name='phase')

		z_shape = [batch_size, latent_dims]
		z_ = tf.placeholder_with_default(tf.random_normal(z_shape), shape=z_shape, name="latent_in")

		slice_sizes = [batch_size, sample_size, residual_channels]
		slice_sizes = [int(x) for x in slice_sizes]
		slice_sizes = tf.constant(slice_sizes, dtype="int32")

		self.source_masked = tf.nn.embedding_lookup(self.input_mask, source_sentence, name = "source_masked")
		self.source_masked_d = tf.slice(self.source_masked, begin=[0,0,0], size=slice_sizes, name = "source_masked_d")

		# Lookup Embeddings and mask embedding beyond source length
		source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence)
		source_embedding = tf.multiply(source_embedding, self.source_masked, name = "source_embedding")

		# Decoder Input
		sample_slice_size = [batch_size, int(options["sample_size"])]
		sample_slice_size = tf.constant(sample_slice_size, dtype="int32")

		# Decoder Output
		target_sentence = tf.slice(target_sentence, [0,1], sample_slice_size, name="target_sentence")

		# Mask loss beyond the target length
		self.target_masked = tf.nn.embedding_lookup(self.output_mask, target_sentence, name="target_masked")

		# Encode Context
		source_dropout = tf.nn.dropout(source_embedding, 0.5)
		encoder_output = self.encoder(source_dropout)

		# Latent distribution parameterized by hidden encoding
		z_mean, z_log_sigma = self.latent_space(encoder_output)

		# Kingma & Welling: only 1 draw necessary as long as minibatch large enough (>100)
		z = self.sample_gaussian(z_mean, z_log_sigma)

		# Process Thoughts Through Memory State
		#context_encoded = self.memory_state(encoder_output, batch_size)

		# Produce Random Thought or Recreate Input
		z = tf.cond(phase, lambda: z, lambda: z_)
		z = tf.reshape(z, [batch_size, sample_size, int(latent_dims / sample_size)])

		tf.summary.histogram("latent_representation", z)

		# Decode Thought
		decoder_output = self.decoder(z)

		total_loss, kl_loss, r_loss, real_kl_loss = self.loss(decoder_output, target_sentence, z_mean, z_log_sigma, kl_weight)
		tf.summary.scalar('loss', total_loss)
		tf.summary.scalar('kl_loss', kl_loss)
		tf.summary.scalar('r_loss', r_loss)
		tf.summary.scalar('real_kl_loss', r_loss)

		flat_logits = tf.reshape(decoder_output, [-1, options['n_target_quant']])
		prediction = tf.argmax(flat_logits, 1)
		
		variables = tf.trainable_variables()
		merged_summary = tf.summary.merge_all()

		tensors = {
			'source_sentence' : source_sentence,
			'target_sentence' : target_sentence,
			'total_loss' : total_loss,
			'r_loss' : r_loss,
			'kl_loss' : kl_loss,
			'real_kl_loss': real_kl_loss,
			'prediction' : prediction,
			'variables' : variables,
			'merged_summary' : merged_summary,
			'source_embedding' : source_embedding,
			'encoder_output' : encoder_output,
			'target_masked' : self.target_masked,
			'source_masked' : self.source_masked,
			'probs' : tf.nn.softmax(flat_logits)
		}

		return tensors

	def latent_space(self, input_):
		"""Latent Space"""

		options = self.options
		input_shape = options["residual_channels"] * options["sample_size"]
		latent_dims = options["latent_dims"]

		input_ = tf.contrib.layers.flatten(input_)

		z_mean = Dense("z_mean", latent_dims)(input_)
		z_log_sigma = Dense("z_log_sigma", latent_dims)(input_)

		return z_mean, z_log_sigma

	def encoder(self, input_):
		"""Utility function for constructing the encoder"""

		options = self.options
		curr_input = input_

		# Connect encoder layers
		for layer_no, dilation in enumerate(self.options['encoder_dilations']):

			layer_output = self.encode_layer(curr_input, dilation, layer_no)

			# Encode only until the input length, conditioning should be 0 beyond that
			layer_output = tf.multiply(layer_output, self.source_masked, name = 'layer_{}_output'.format(layer_no))

			curr_input = layer_output
		
		# What is this?
		processed_output = conv1d(
			tf.nn.relu(layer_output), 
			options['residual_channels'], 
			name='encoder_post_processing'
		)

		# What are these?
		processed_output = tf.nn.relu(processed_output)
		processed_output = tf.multiply(processed_output, self.source_masked_d, name='encoder_processed')

		return processed_output

	def memory_state(self, input_, batch_size):
		"""Create the memory state and feed encoded thoughts
		throught it"""

		options = self.options

		# Feed encoded thoughts into memory cell
		lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(options["sample_size"])

		# Run thoughts through the cell
		options = {
			"dtype": tf.float32
		}

		output, output_state = tf.contrib.rnn.static_rnn(lstm, tf.unstack(input_), **options)

		last_output = tf.gather(output, batch_size)

		return tf.expand_dims(last_output, 0)

	def decoder(self, input_):
		"""Utility function for constructing the decoder"""

		options = self.options
		batch_size = options["batch_size"]
		latent_dims = options["latent_dims"]
		sample_size = options["sample_size"]
		reshape_dims = [batch_size, sample_size, int(latent_dims / sample_size)]
		curr_input = input_

		for layer_no, dilation in enumerate(options['decoder_dilations']):
			layer_output = self.decode_layer(curr_input, dilation, layer_no)
			curr_input = layer_output

		processed_output = conv1d(
			tf.nn.relu(layer_output), 
			options['n_target_quant'], 
			name="decoder_post_processing"
		)

		tf.summary.histogram("final_conv", processed_output)

		return processed_output

	def encode_layer(self, input_, dilation, layer_no, last_layer=False):
		"""Utility function for forming an encode layer"""

		options = self.options

		# Reduce Dimension
		normed = tf.contrib.layers.layer_norm(input_)
		relu1 = tf.nn.relu(normed, name='enc_relu1_layer{}'.format(layer_no))
		conv1 = conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))

		# What is this?
		conv1 = tf.multiply(conv1, self.source_masked_d)
		
		# Unmasked 1 x k dilated convolution
		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = dilated_conv1d(relu2, options['residual_channels'], 
			dilation, options['encoder_filter_width'],
			causal = False, 
			name = "enc_dilated_conv_layer{}".format(layer_no)
		)

		# What is this?
		dilated_conv = tf.multiply(dilated_conv, self.source_masked_d)

		# Restore Dimension
		relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu3_layer{}'.format(layer_no))
		conv2 = conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))

		# Residual connection
		return input_ + conv2

	def decode_layer(self, input_, dilation, layer_no):
		"""Utility function for forming a decode layer"""

		options = self.options

		# Input dimension
		in_dim = input_.get_shape().as_list()[-1]

		# Reduce dimension
		normed = tf.contrib.layers.layer_norm(input_)
		relu1 = tf.nn.relu(input_, name = 'dec_relu1_layer{}'.format(layer_no))
		conv1 = conv1d(relu1, in_dim, name = 'dec_conv1d_1_layer{}'.format(layer_no))

		# Masked 1 x k dilated convolution
		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = dilated_conv1d(
			relu2,
			output_channels = in_dim,
			dilation        = dilation,
			filter_width    = options['decoder_filter_width'],
			causal          = True,
			name            = "dec_dilated_conv_layer{}".format(layer_no))

		# Restore dimension
		relu3 = tf.nn.relu(dilated_conv, name = 'dec_relu3_layer{}'.format(layer_no))
		conv2 = conv1d(relu3, in_dim, name = 'dec_conv1d_2_layer{}'.format(layer_no))

		# Residual connection
		return input_ + conv2

	def kullback_leibler(self, mu, log_sigma):
		"""(Gaussian) Kullback-Leibler divergence"""

		with tf.name_scope("KL_divergence"):
			KL = 1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma)
			KL = -0.5 * tf.reduce_sum(KL, 1)
			tf.summary.histogram("KL", KL)
			return KL

	def sample_gaussian(self, mu, log_sigma):
		"""Draw sample from Gaussian with given shape, subject 
        to random noise epsilon"""

		with tf.name_scope("sample_gaussian"):
			epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
			return mu + epsilon * tf.exp(log_sigma) 

	def loss(self, decoder_output, target_sentences, z_mean, z_log_sigma, kl_weight):
		"""Calculate loss between decoder output and target"""

		options = self.options

		target_one_hot = tf.one_hot(
			target_sentences, 
			depth=options['n_target_quant'], 
			dtype=tf.float32
		)

		# Calculate Loss
		loss = tf.nn.softmax_cross_entropy_with_logits(
			logits=decoder_output,
			labels=target_one_hot, 
			name='decoder_cross_entropy_loss'
		)

		# Add KL Loss
		kl_loss = self.kullback_leibler(z_mean, z_log_sigma)

		# Mask loss beyond EOL in target
		if 'target_mask_chars' in options:
			masked_target = tf.squeeze(self.target_masked, 2)
			r_loss = tf.multiply(loss, masked_target, name='masked_loss')
			r_loss = tf.reduce_sum(r_loss, 1)
		else:
			r_loss = tf.reduce_sum(r_loss, 1, name="Reduced_mean_loss")

		average_kl_loss = tf.reduce_mean(kl_loss)
		kl_loss = tf.multiply(kl_weight, kl_loss)
		weighted_kl_loss = tf.reduce_mean(kl_loss)
		average_r_loss = tf.reduce_mean(r_loss)
		total_loss = tf.reduce_mean(r_loss + kl_loss, name="cost")

		return total_loss, weighted_kl_loss, average_r_loss, average_kl_loss

	def masked_loss(self, decoder_output, target_sentences, z_mean, z_log_sigma, kl_weight):
		"""Calculate loss between decoder output and target"""

		options = self.options

		target_one_hot = tf.one_hot(
			target_sentences, 
			depth=options['n_target_quant'], 
			dtype=tf.float32
		)

		# Calculate Loss
		loss = tf.nn.softmax_cross_entropy_with_logits(
			logits=decoder_output,
			labels=target_one_hot, 
			name='decoder_cross_entropy_loss'
		)

		# Add KL Loss
		kl_loss = self.kullback_leibler(z_mean, z_log_sigma)

		# Mask KL Loss
		masked_target = tf.squeeze(self.target_masked, 2)
		target_lengths = tf.reduce_sum(masked_target, 1)
		full_target = tf.reduce_sum(tf.ones(tf.shape(masked_target)), 1)
		kl_multiplier = tf.div(target_lengths, full_target)
		kl_loss = tf.multiply(kl_multiplier, kl_loss)
		kl_loss = tf.multiply(kl_weight, kl_loss)

		# Mask loss beyond EOL in target
		if 'target_mask_chars' in options:
			r_loss = tf.multiply(loss, masked_target, name='masked_loss')
			r_loss = tf.reduce_sum(r_loss, 1)
			r_loss = tf.div(r_loss, target_lengths, name="Reduced_mean_loss")
		else:
			r_loss = tf.reduce_sum(r_loss, 1, name="Reduced_mean_loss")

		average_kl_loss = tf.reduce_mean(kl_loss)
		average_r_loss = tf.reduce_mean(r_loss)
		total_loss = tf.reduce_mean(r_loss + kl_loss, name="cost")

		return total_loss, average_kl_loss, average_r_loss

class Dense():

	def __init__(self, scope="dense_layer", size=None, dropout=1., nonlinearity=tf.identity):

		assert size, "Must specify layer size (num nodes)"

		self.scope = scope
		self.size = size
		self.dropout = dropout
		self.nonlinearity = nonlinearity

	def __call__(self, x):
		with tf.name_scope(self.scope):
			while True:
				try:
					#x = tf.contrib.layers.layer_norm(x)
					output = self.nonlinearity(tf.matmul(x, self.w) + self.b)
					return output
				except(AttributeError):
					value = x.get_shape()[1].value
					self.w, self.b = self.wb_vars(value, self.size)
					self.w = tf.nn.dropout(self.w, self.dropout)

	@staticmethod
	def wb_vars(fan_in: int, fan_out: int):
		"""Helper to initialize weights and biases using
        Xavier initialization for ReLUs"""

		stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

		initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
		initial_b = tf.zeros([fan_out])

		return (tf.Variable(initial_w, trainable=True, name="weights"),
				tf.Variable(initial_b, trainable=True, name="biases"))

# Utility Functions and Classes 

def conv1d(input_, output_channels, filter_width=1, stride=1, stddev=0.02, name='conv1d'):
	"""Helper function to create and store weights and biases with convolutional layer"""

	with tf.variable_scope(name):

		input_shape = input_.get_shape()
		input_channels = input_shape[-1]
		shape = [filter_width, input_channels, output_channels]

		weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
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
		weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
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
	"""Helper function for appling masked convolutions"""
	
	# Padding for Masked Convolutions
	if causal:
		padding = [[0, 0], [int((filter_width - 1) * dilation), 0], [0, 0]]
		padded = tf.pad(input_, padding)
	else:
		middle_shape = [int((filter_width - 1) * dilation/2), int((filter_width - 1) * dilation/2)]
		padding = [[0, 0], middle_shape, [0, 0]]
		padded = tf.pad(input_, padding)
	
	d_conv = atrous_conv1d(padded, output_channels, filter_width=filter_width, rate=dilation, name=name)	
	result = tf.slice(d_conv,[0, 0, 0],[-1, int(input_.get_shape()[1]), -1])
	
	return result