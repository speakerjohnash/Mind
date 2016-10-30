#!/usr/local/bin/python3.3

"""This module loads classifier from various libraries and produces 
helper functions that will classify thoughts. Depending on the model 
requested this module will load a different previously generated model.

Created on Apr 27, 2016
@author: Matthew Sevrens
"""

from os.path import isfile
import sys
import logging
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from mind.tools import load_json
from mind.tensorflow_cnn import validate_config, get_tensor, string_to_tensor
from mind.bilstm_tagger import validate_config as bilstm_validate_config
from mind.bilstm_tagger import doc_to_tensor

def get_tf_cnn_by_name(model_name, gpu_mem_fraction=False):
	"""Load a tensorFlow CNN by name"""

	base = "models/"
	# Switch on Models
	if model_name == "thought_type":
		model_path = base + "thought_type.ckpt"
		label_map_path = {"0": "Predict", "1": "State", "2": "Ask", "3": "Reflect"}
	elif model_name == "sentiment":
		model_path = base + "sentiment.ckpt"
		label_map_path = {"0": "Negative", "1": "Positive"}
	else:
		logging.warning("Model not found. Terminating")
		sys.exit()

	return get_tf_cnn_by_path(model_path, label_map_path, gpu_mem_fraction=gpu_mem_fraction)

def get_tf_cnn_by_path(model_path, label_map_path, gpu_mem_fraction=False):
	"""Load a tensorFlow module by name"""

	# Load Config
	config_path = "config/tf_cnn_config.json"

	# Validate Model and Label Map
	if not isfile(model_path):
		logging.warning("Resouces to load model not found. Loading from S3")
		sys.exit()

	# Load Config
	config = load_json(config_path)
	config["label_map"] = label_map_path
	config["model_path"] = model_path
	meta_path = model_path.split(".ckpt")[0] + ".meta"
	config = validate_config(config)
	label_map = config["label_map"]

	# Load Session and Graph
	ops.reset_default_graph()
	saver = tf.train.import_meta_graph(meta_path)
	sess = tf.Session()
	saver.restore(sess, config["model_path"])
	graph = sess.graph
	model = get_tensor(graph, "model:0")
	
	# Generate Helper Function
	def apply_cnn(thoughts, doc_key="Thought", label_key="CNN"):
		"""Apply CNN to thoughts"""

		alphabet_length = config["alphabet_length"]
		doc_length = config["doc_length"]
		batch_size = len(thoughts)

		tensor = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))

		for index, doc in enumerate(thoughts):
			tensor[index][0] = string_to_tensor(config, doc[doc_key], doc_length)

		tensor = np.transpose(tensor, (0, 1, 3, 2))
		feed_dict_test = {get_tensor(graph, "x:0"): tensor}
		output = sess.run(model, feed_dict=feed_dict_test)
		labels = output[:,1] if "sentiment" in model_path else np.argmax(output, 1)

		if "sentiment" in model_path:
			for index, thought in enumerate(thoughts):
				thought[label_key] = math.pow(10, labels[index])				
		else:
			for index, thought in enumerate(thoughts):
				label = label_map.get(str(labels[index]), "")
				thought[label_key] = label

		return thoughts

	return apply_cnn

def get_tf_rnn_by_path(model_path, w2i_path, gpu_mem_fraction=False, model_name=False):
	"""Load a tensorflow rnn model"""

	config_path = "config/bilstm_config.json"

	if not isfile(model_path):
		logging.warning("Resources to load model not found.")
		sys.exit()

	# Load Graph
	config = bilstm_validate_config(config_path)
	config["model_path"] = model_path
	meta_path = model_path.split(".ckpt")[0] + ".meta"
	config["w2i"] = load_params(w2i_path)

	# Load Session and Graph
	ops.reset_default_graph()
	saver = tf.train.import_meta_graph(meta_path)

	if gpu_mem_fraction:
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
	else:
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	saver.restore(sess, config["model_path"])
	graph = sess.graph

	if not model_name:
		model = get_tensor(graph, "model:0")
	else:
		model = get_tensor(graph, model_name)

	# Generate Helper Function
	def apply_rnn(thought, doc_key="thought", label_key="tag"):

		for index, doc in enumerate(thought):
			thot = doc[doc_key].lower().split()[0:config["max_tokens"]]
			char_inputs, word_lengths, word_indices, _ = doc_to_tensor(config, sess, graph, tran)
			feed_dict = {
				get_tensor(graph, "char_inputs:0"): char_inputs,
				get_tensor(graph, "word_inputs:0"): word_indices,
				get_tensor(graph, "word_lengths:0"): word_lengths,
				get_tensor(graph, "doc_length:0"): len(thot),
				get_tensor(graph, "train:0"): False
			}

			output = sess.run(model, feed_dict=feed_dict)
			output = [config["tag_map"][str(i)] for i in np.argmax(output, 1)]
			target_indices = [i for i in range(len(output)) if output[i] == "target"]
			doc[label_key] = " ".join([tran[i] for i in target_indices])

		return thought

	return apply_rnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
