#!/usr/local/bin/python3
# pylint: disable=unused-variable
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Mar 26, 2016
@author: Matthew Sevrens
"""

############################################# USAGE ###############################################

# python3 -m mind.tensorflow_cnn [config]
# python3 -m mind.tensorflow_cnn config/tf_cnn_config.json

# For addtional details on implementation see:
# Character-level Convolutional Networks for Text Classification
# http://arxiv.org/abs/1509.01626
#
# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# http://arxiv.org/abs/1502.03167

###################################################################################################

import logging
import math
import os
import pprint
import random
import shutil
import sys

import pandas as pd
import numpy as np
import tensorflow as tf

from mind.tools import load_json

logging.basicConfig(level=logging.INFO)

def chunks(array, num):
	"""Chunk array into equal sized parts"""
	num = max(1, num)
	return [array[i:i + num] for i in range(0, len(array), num)]

def validate_config(config):
	"""Validate input configuration"""

	config = load_json(config)
	logging.debug("Configuration is :\n{0}".format(pprint.pformat(config)))
	reshape = ((config["doc_length"] - 78) / 27) * 256
	config["reshape"] = int(reshape)
	config["label_map"] = load_json(config["label_map"])
	config["num_labels"] = len(config["label_map"].keys())
	config["alpha_dict"] = {a : i for i, a in enumerate(config["alphabet"])}
	config["base_rate"] = config["base_rate"] * math.sqrt(config["batch_size"]) / math.sqrt(128)
	config["alphabet_length"] = len(config["alphabet"])

	return config

def load_labeled_data(config):
	"""Load labeled data and label map"""

	dataset = config["dataset"]
	label_map = config["label_map"]
	label_key = config["label_key"]

	reversed_map = dict(zip(label_map.values(), label_map.keys()))
	map_labels = lambda x: reversed_map.get(str(x[label_key]), "")

	df = pd.read_csv(dataset, na_filter=False, encoding="utf-8", error_bad_lines=False)
	df["LABEL_NUM"] = df.apply(map_labels, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	grouped_train = train.groupby('LABEL_NUM', as_index=False)
	groups_train = dict(list(grouped_train))

	return train, test, groups_train

def mixed_batching(config, df, groups_train):
	"""Batch from train data using equal class batching"""

	num_labels = config["num_labels"]
	batch_size = config["batch_size"]
	half_batch = int(batch_size / 2)
	indices_to_sample = list(np.random.choice(df.index, half_batch))

	for index in range(half_batch):
		label = random.randint(0, num_labels - 1)
		select_group = groups_train[str(label)]
		indices_to_sample.append(np.random.choice(select_group.index, 1)[0])

	random.shuffle(indices_to_sample)
	batch = df.loc[indices_to_sample]

	return batch

def batch_to_tensor(config, batch):
	"""Convert a batch to a tensor representation"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	num_labels = config["num_labels"]
	batch_size = len(batch.index)

	labels = np.array(batch["LABEL_NUM"].astype(int))
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	docs = batch["Thought"].tolist()
	encoded_thoughts = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))
	
	for index, thoughts in enumerate(docs):
		encoded_thoughts[index][0] = string_to_tensor(config, thoughts, doc_length)

	encoded_thoughts = np.transpose(encoded_thoughts, (0, 1, 3, 2))
	return encoded_thoughts, labels

def string_to_tensor(config, doc, length):
	"""Convert thought to tensor format"""
	alphabet = config["alphabet"]
	alpha_dict = config["alpha_dict"]
	doc = doc.lower()[0:length]
	tensor = np.zeros((len(alphabet), length), dtype=np.float32)
	for index, char in reversed(list(enumerate(doc))):
		if char in alphabet:
			tensor[alpha_dict[char]][len(doc) - index - 1] = 1
	return tensor

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	total_count = len(test.index)
	correct_count = 0
	chunked_test = chunks(np.array(test.index), 128)
	num_chunks = len(chunked_test)

	for i in range(num_chunks):

		batch_test = test.loc[chunked_test[i]]
		batch_size = len(batch_test)

		thoughts_test, labels_test = batch_to_tensor(config, batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): thoughts_test}
		output = sess.run(model, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count
	
	test_accuracy = 100.0 * (correct_count / total_count)
	logging.info("Test accuracy: %.2f%%" % test_accuracy)
	logging.info("Correct count: " + str(correct_count))
	logging.info("Total count: " + str(total_count))

	return test_accuracy
	
def accuracy(predictions, labels):
	"""Return accuracy for a batch"""
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def get_tensor(graph, name):
	"""Get tensor by name"""
	return graph.get_tensor_by_name(name)

def get_op(graph, name):
	"""Get operation by name"""
	return graph.get_operation_by_name(name)

def get_variable(graph, name):
	"""Get variable by name"""
	with graph.as_default():
		variable = [v for v in tf.all_variables() if v.name == name][0]
		return variable

def threshold(tensor):
	"""ReLU with threshold at 1e-6"""
	return tf.mul(tf.to_float(tf.greater_equal(tensor, 1e-6)), tensor)

def bias_variable(shape, flat_input_shape):
	"""Initialize biases"""
	stdv = 1 / math.sqrt(flat_input_shape)
	bias = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), name="B")
	return bias

def weight_variable(config, shape):
	"""Initialize weights"""
	weight = tf.Variable(tf.mul(tf.random_normal(shape), config["randomize"]), name="W")
	return weight

def conv2d(input_x, weights):
	"""Create convolutional layer"""
	layer = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='VALID')
	return layer

def max_pool(tensor):
	"""Create max pooling layer"""
	layer = tf.nn.max_pool(tensor, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
	return layer

def build_graph(config):
	"""Build CNN"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]
	base_rate = config["base_rate"]
	batch_size = config["batch_size"]
	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(base_rate, trainable=False, name="lr")

		input_shape = [None, 1, doc_length, alphabet_length]
		output_shape = [None, num_labels]

		thoughts_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y")

		# Encoder Weights and Biases
		w_conv1 = weight_variable(config, [1, 7, alphabet_length, 256])
		b_conv1 = bias_variable([256], 7 * alphabet_length)

		w_conv2 = weight_variable(config, [1, 7, 256, 256])
		b_conv2 = bias_variable([256], 7 * 256)

		w_conv3 = weight_variable(config, [1, 3, 256, 256])
		b_conv3 = bias_variable([256], 3 * 256)

		w_conv4 = weight_variable(config, [1, 3, 256, 256])
		b_conv4 = bias_variable([256], 3 * 256)

		w_conv5 = weight_variable(config, [1, 3, 256, 256])
		b_conv5 = bias_variable([256], 3 * 256)

		w_fc1 = weight_variable(config, [reshape, 1024])
		b_fc1 = bias_variable([1024], reshape)

		w_fc2 = weight_variable(config, [1024, num_labels])
		b_fc2 = bias_variable([num_labels], 1024)

		bn_scaler = tf.Variable(1.0 * tf.ones([num_labels]))

		# Utility for Batch Normalization
		layer_sizes = [256] * 8 + [1024, num_labels]
		ewma = tf.train.ExponentialMovingAverage(decay=0.99)
		running_mean = [tf.Variable(tf.zeros([l]), trainable=False) for l in layer_sizes]
		running_var = [tf.Variable(tf.ones([l]), trainable=False) for l in layer_sizes]
		bn_assigns = []

		def layer(input_h, details, layer_type, train, weights=None, biases=None):
			"""Apply all necessary steps in a layer"""

			# Preactivation
			if layer_type == "conv":
				z_pre = conv2d(input_h, weights)
			elif layer_type == "pool":
				z_pre = max_pool(input_h)
			elif layer_type == "fc":
				z_pre = tf.matmul(input_h, weights)

			details["layer_count"] += 1
			layer_n = details["layer_count"]

			if train:
				z = update_batch_normalization(z_pre, layer_n)
			else:
				mean = ewma.average(running_mean[layer_n-1])
				var = ewma.average(running_var[layer_n-1])
				z = batch_normalization(z_pre, mean=mean, var=var)

			# Apply Activation
			if layer_type == "conv" or layer_type == "fc":
				layer = threshold(z + biases)
			else:
				layer = z

			return layer

		def batch_normalization(batch, mean=None, var=None):
			"""Perform batch normalization"""
			if mean == None or var == None:
				axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
				mean, var = tf.nn.moments(batch, axes=axes)
			return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

		def update_batch_normalization(batch, l):
			"batch normalize + update average mean and variance of layer l"
			axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
			mean, var = tf.nn.moments(batch, axes=axes)
			assign_mean = running_mean[l-1].assign(mean)
			assign_var = running_var[l-1].assign(var)
			bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
			with tf.control_dependencies([assign_mean, assign_var]):
				return (batch - mean) / tf.sqrt(var + 1e-10)

		def encoder(inputs, name, train=False, noise_std=0.0):
			"""Add model layers to the graph"""

			details = {"layer_count": 0}

			h_conv1 = layer(inputs, details, "conv", train, weights=w_conv1, biases=b_conv1)
			h_pool1 = layer(h_conv1, details, "pool", train)

			h_conv2 = layer(h_pool1, details, "conv", train, weights=w_conv2, biases=b_conv2)
			h_pool2 = layer(h_conv2, details, "pool", train)

			h_conv3 = layer(h_pool2, details, "conv", train, weights=w_conv3, biases=b_conv3)

			h_conv4 = layer(h_conv3, details, "conv", train, weights=w_conv4, biases=b_conv4)

			h_conv5 = layer(h_conv4, details, "conv", train, weights=w_conv5, biases=b_conv5)
			h_pool5 = layer(h_conv5, details, "pool", train)

			h_reshape = tf.reshape(h_pool5, [-1, reshape])

			h_fc1 = layer(h_reshape, details, "fc", train, weights=w_fc1, biases=b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = layer(h_fc1, details, "fc", train, weights=w_fc2, biases=b_fc2)

			softmax = tf.nn.softmax(bn_scaler * h_fc2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = encoder(thoughts_placeholder, "network", train=True)
		trained_model = encoder(thoughts_placeholder, "model", train=False)

		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels_placeholder, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies([optimizer]):
			bn_applier = tf.group(bn_updates, name="bn_applier")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	epochs = config["epochs"]
	eras = config["eras"]
	dataset = config["dataset"]
	train, test, groups_train = load_labeled_data(config)
	num_eras = epochs * eras
	logging_interval = 50
	learning_rate_interval = 15000

	best_accuracy, best_era = 0, 0
	save_dir = "models/checkpoints/"
	os.makedirs(save_dir, exist_ok=True)
	checkpoints = {}

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		thoughts, labels = batch_to_tensor(config, batch)
		feed_dict = {get_tensor(graph, "x:0") : thoughts, get_tensor(graph, "y:0") : labels}

		# Run Training Step
		sess.run(get_op(graph, "optimizer"), feed_dict=feed_dict)
		sess.run(get_op(graph, "bn_applier"), feed_dict=feed_dict)

		# Log Loss
		if step % logging_interval == 0:
			loss = sess.run(get_tensor(graph, "loss:0"), feed_dict=feed_dict)
			logging.info("train loss at epoch %d: %g" % (step + 1, loss))

		# Log Accuracy for Tracking
		if step % 1000 == 0:
			predictions = sess.run(get_tensor(graph, "model:0"), feed_dict=feed_dict)
			logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))

		# Evaluate Testset, Log Progress and Save
		if step != 0 and step % epochs == 0:

			#Evaluate Model
			model = get_tensor(graph, "model:0")
			learning_rate = get_variable(graph, "lr:0")
			logging.info("Testing for era %d" % (step / epochs))
			logging.info("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))
			test_accuracy = evaluate_testset(config, graph, sess, model, test)

			# Save Checkpoint
			current_era = int(step / epochs)
			meta_path = save_dir + "era_" + str(current_era) + ".ckpt.meta"
			model_path = saver.save(sess, save_dir + "era_" + str(current_era) + ".ckpt")
			logging.info("Checkpoint saved in file: %s" % model_path)
			checkpoints[current_era] = model_path

			# Stop Training if Converged
			if test_accuracy > best_accuracy:
				best_era = current_era
				best_accuracy = test_accuracy

			if current_era - best_era == 3:
				model_path = checkpoints[best_era]
				break

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Clean Up Directory
	dataset_path = os.path.basename(dataset).split(".")[0]
	final_model_path = "models/" + dataset_path + ".ckpt"
	final_meta_path = "models/" + dataset_path + ".meta"
	logging.info("Moving final model from {0} to {1}.".format(model_path, final_model_path))
	os.rename(model_path, final_model_path)
	os.rename(meta_path, final_meta_path)
	logging.info("Deleting unneeded directory of checkpoints at {0}".format(save_dir))
	shutil.rmtree(save_dir)

	return final_model_path

def run_session(config, graph, saver):
	"""Run Session"""

	with tf.Session(graph=graph) as sess:

		mode = config["mode"]
		model_path = config.get("model_path", "")

		tf.initialize_all_variables().run()

		if mode == "train":
			train_model(config, graph, sess, saver)
		elif mode == "test":
			saver.restore(sess, model_path)
			model = get_tensor(graph, "model:0")
			_, test, _ = load_data(config)
			evaluate_testset(config, graph, sess, model, test)

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()