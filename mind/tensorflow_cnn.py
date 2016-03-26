#!/usr/local/bin/python3
# pylint: disable=unused-variable
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Mar 26, 2016
@author: Matthew Sevrens
"""

#################### USAGE #######################

# python3 -m mind.tensorflow_cnn [config]
# python3 -m mind.tensorflow_cnn config/tf_cnn_config.json

# For details on implementation see:
# Character-level Convolutional Networks for Text Classification
# http://arxiv.org/abs/1509.01626

##################################################

import csv
import logging
import math
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from meerkat.various_tools import load_piped_dataframe
from meerkat.classification.tools import fill_description_unmasked, reverse_map
from meerkat.classification.verify_data import load_json

def chunks(array, num):
	"""Chunk array into equal sized parts"""
	num = max(1, num)
	return [array[i:i + num] for i in range(0, len(array), num)]

def validate_config(filename):
	"""Validate input configuration"""

	default_config = "config/default_tf_config.json"
	config = load_json(filename)
	reshape = ((config["doc_length"] - 96) / 27) * 256
	config["alpha_dict"] = {a : i for i, a in enumerate(config["alphabet"])}
	config["label_map"] = load_json(config["label_map"])
	config["num_labels"] = len(config["label_map"].keys())
	config["base_rate"] = config["base_rate"] * math.sqrt(config["batch_size"]) / math.sqrt(128)
	config["alphabet_length"] = len(config["alphabet"])

	if reshape.is_integer():
		config["reshape"] = int(reshape)
	else:
		raise ValueError('DOC_LENGTH - 96 must be divisible by 27: 123, 150, 177, 204...')

	return config

def load_data(config):
	"""Load data and label map"""

	print("Load data here")


def evaluate_testset(config, graph, sess, model, test, chunked_test):
	"""Check error on test set"""

	batch_size = config["batch_size"]
	total_count = 0
	correct_count = 0
	num_chunks = len(chunked_test)

	for i in range(num_chunks):

		batch_test = test.loc[chunked_test[i]]
		batch_length = len(batch_test)
		if batch_length != 128:
			continue

		trans_test, labels_test = batch_to_tensor(config, batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
		output = sess.run(model, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count
		total_count += batch_size
	
	test_accuracy = 100.0 * (correct_count / total_count)
	logging.warning("Test accuracy: %.2f%%" % test_accuracy)
	logging.warning("Correct count: " + str(correct_count))

def mixed_batching(config, df, groups_train):
	"""Batch from train data using equal class batching"""

	num_labels = config["num_labels"]
	batch_size = config["batch_size"]
	half_batch = int(batch_size / 2)
	indices_to_sample = list(np.random.choice(df.index, half_batch))

	for index in range(half_batch):
		label = random.randint(1, num_labels)
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
	batch_size = config["batch_size"]

	labels = np.array(batch["LABEL_NUM"].astype(int)) - 1
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	docs = batch["DESCRIPTION_UNMASKED"].tolist()
	transactions = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))
	
	for index, trans in enumerate(docs):
		transactions[index][0] = string_to_tensor(config, trans, doc_length)

	transactions = np.transpose(transactions, (0, 1, 3, 2))
	return transactions, labels

def string_to_tensor(config, doc, length):
	"""Convert transaction to tensor format"""
	alphabet = config["alphabet"]
	alpha_dict = config["alpha_dict"]
	doc = doc.lower()[0:length]
	tensor = np.zeros((len(alphabet), length), dtype=np.float32)
	for index, char in reversed(list(enumerate(doc))):
		if char in alphabet:
			tensor[alpha_dict[char]][len(doc) - index - 1] = 1
	return tensor
	
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

def accuracy(predictions, labels):
	"""Return accuracy for a batch"""
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def threshold(tensor):
	"""ReLU with threshold at 1e-6"""
	return tf.mul(tf.to_float(tf.greater_equal(tensor, 1e-6)), tensor)

def softmax_with_temperature(tensor, temperature):
	"""Softmax with temperature variable"""
	return tf.div(tf.exp(tensor/temperature), tf.reduce_sum(tf.exp(tensor/temperature)))

def bias_variable(shape, mult):
	"""Initialize biases"""
	stdv = 1 / math.sqrt(mult)
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
	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(base_rate, trainable=False, name="lr") 

		input_shape = [None, 1, doc_length, alphabet_length]
		output_shape = [None, num_labels]

		trans_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y")

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

		def model(data, name, train=False):
			"""Add model layers to the graph"""

			h_conv1 = threshold(conv2d(data, w_conv1) + b_conv1)
			h_pool1 = max_pool(h_conv1)

			h_conv2 = threshold(conv2d(h_pool1, w_conv2) + b_conv2)
			h_pool2 = max_pool(h_conv2)

			h_conv3 = threshold(conv2d(h_pool2, w_conv3) + b_conv3)

			h_conv4 = threshold(conv2d(h_conv3, w_conv4) + b_conv4)

			h_conv5 = threshold(conv2d(h_conv4, w_conv5) + b_conv5)
			h_pool5 = max_pool(h_conv5)

			h_reshape = tf.reshape(h_pool5, [-1, reshape])

			h_fc1 = threshold(tf.matmul(h_reshape, w_fc1) + b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2

			softmax = tf.nn.softmax(h_fc2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = model(trans_placeholder, "network", train=True)
		trained_model = model(trans_placeholder, "model", train=False)

		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels_placeholder, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	epochs = config["epochs"]
	eras = config["eras"]
	dataset = config["dataset"]
	train, test, groups_train, chunked_test = load_data(config)
	num_eras = epochs * eras
	logging_interval = 50
	learning_rate_interval = 15000

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		trans, labels = batch_to_tensor(config, batch)
		feed_dict = {get_tensor(graph, "x:0") : trans, get_tensor(graph, "y:0") : labels}

		# Run Training Step
		sess.run(get_op(graph, "optimizer"), feed_dict=feed_dict)

		# Log Loss
		if step % logging_interval == 0:
			loss = sess.run(get_tensor(graph, "loss:0"), feed_dict=feed_dict)
			logging.warning("train loss at epoch %d: %g" % (step + 1, loss))

		# Evaluate Testset and Log Progress
		if step != 0 and step % epochs == 0:
			model = get_tensor(graph, "model:0")
			learning_rate = get_variable(graph, "lr:0")
			predictions = sess.run(model, feed_dict=feed_dict)
			logging.warning("Testing for era %d" % (step / epochs))
			logging.warning("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))
			logging.warning("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))
			evaluate_testset(config, graph, sess, model, test, chunked_test)

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Save Model
	save_dir = "meerkat/classification/models/"
	save_path = saver.save(sess, save_dir + "model_" + dataset.split(".")[0] + ".ckpt")
	logging.warning("Model saved in file: %s" % save_path)

def run_session(config, graph, saver):
	"""Run Session"""

	with tf.Session(graph=graph) as sess:

		mode = config["mode"]
		model_path = config["model_path"]

		tf.initialize_all_variables().run()

		if mode == "train":
			train_model(config, graph, sess, saver)
		elif mode == "test":
			saver.restore(sess, model_path)
			model = get_tensor(graph, "model:0")
			_, test, _, chunked_test = load_data(config)
			evaluate_testset(config, graph, sess, model, test, chunked_test)

def run_from_command_line():
	"""Run module from command line"""
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
