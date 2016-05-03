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

import numpy as np
import tensorflow as tf

from mind.tools import load_json
from mind.tensorflow_cnn import validate_config, get_tensor, string_to_tensor, build_graph

def get_tf_cnn_by_name(model_name, gpu_mem_fraction=False):
	"""Load a tensorFlow CNN by name"""

	base = "models/"
	# Switch on Models
	if model_name == "thought_type":
		model_path = base + "thought_type.ckpt"
		label_map_path = {"0": "Predict", "1": "State", "2": "Ask", "3": "Reflect"},
	elif model_name == "sentiment":
		model_path = base + "sentiment.ckpt"
		label_map_path = {"0": "Positive", "1": "Negative"}
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
	config = validate_config(config)
	label_map = config["label_map"]

	# Load Model
	graph, saver = build_graph(config)
	sess = tf.Session(graph=graph)
	saver.restore(sess, config["model_path"])
	model = get_tensor(graph, "model:0")
	
	# Generate Helper Function
	def apply_cnn(thoughts, doc_key="description", label_key="CNN", label_only=True):
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
		labels = np.argmax(output, 1)
	
		for index, thought in enumerate(thoughts):
			label = label_map.get(str(labels[index]), "")
			thought[label_key] = label

		return thoughts

	return apply_cnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
