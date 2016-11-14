#!/usr/local/bin/python3

"""Train a bi-LSTM tagger using tensorFlow

Created on Aug 15, 2016
@author: Matthew Sevrens
"""

############################################# USAGE ###############################################

# python3 -m mind.bilstm_tagger [config_file]
# python3 -m mind.bilstm_tagger config/bilstm_config.json

# For addtional details on implementation see:
#
# Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss
# http://arxiv.org/pdf/1604.05529v2.pdf
# https://github.com/bplank/bilstm-aux
#
# Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation
# http://arxiv.org/pdf/1508.02096v2.pdf

###################################################################################################

import logging
import math
import os
import pprint
import random
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from mind.tools import get_tensor, get_op, get_variable
from mind.tools import load_json, load_piped_dataframe, reverse_map

logging.basicConfig(level=logging.INFO)

def last_relevant(output, length, name):
	batch_size = tf.shape(output)[0]
	max_length = tf.reduce_max(tf.to_int32(length))
	output_size = int(output.get_shape()[2])
	index = tf.range(0, batch_size) * max_length + (tf.to_int32(length) - 1)
	flat = tf.reshape(output, [-1, output_size])
	relevant = tf.gather(flat, index, name=name)
	return relevant

def get_tags(config, doc):
	"""Convert df row to list of tags and tokens"""

	tokens = doc["thought"].lower().split()[0:config["max_tokens"]]
	tag = doc["tag"].lower().split()
	tags = []

	for token in tokens:
		found = False
		for word in tag:
			if word in token and tag.index(word) == 0:
				tags.append("target")
				tag = tag[1:]
				found = True
				break
		if not found:
			tags.append("background")
	return (tokens, tags)

def validate_config(config):
	"""Validate input configuration"""

	config = load_json(config)
	config["c2i"] = {a : i + 3 for i, a in enumerate(config["alphabet"])}
	config["c2i"]["_UNK"] = 0
	config["c2i"]["<w>"] = 1
	config["c2i"]["</w>"] = 2

	return config

def load_data(config):
	"""Load labeled data"""

	df = load_piped_dataframe(config["dataset"])
	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	return train, test

def load_embeddings_file(file_name, sep=" ",lower=False):
	"""Load embeddings file"""

	emb = {}

	for line in open(file_name):
		fields = line.split(sep)
		word = fields[0]
		if lower:
			word = word.lower()
		emb[word] = [float(x) for x in fields[1:]]

	logging.info("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
	return emb, len(emb[word])

def words_to_indices(data):
	"""convert tokens to int, assuming data is a df"""
	w2i = {}
	w2i["_UNK"] = 0
	for row_num, row in enumerate(data):
		for token in row[0]:
			if token not in w2i:
				w2i[token] = len(w2i)
	return w2i

def construct_embedding(config, w2i, loaded_embedding):
	"""construct an embedding that contains all words in loaded_embedding and w2i"""
	num_words = len(set(loaded_embedding.keys()).union(set(w2i.keys())))
	# initialize a num_words * we_dim embedding table
	temp = np.random.uniform(-1,1, (num_words, config["we_dim"]))
	for word in loaded_embedding.keys():
		if word not in w2i:
			w2i[word] = len(w2i)
		temp[w2i[word]] = loaded_embedding[word]
	return w2i, temp

def subpreprocess(config, name):
	config[name] = config[name].to_dict("record")
	for i, tran in enumerate(config[name]):
		config[name][i] = get_tags(config, tran)
	return config

def preprocess(config):
	"""Split data into training and test, return w2i for data in training, return training data's embedding matrix"""
	config["train"], config["test"] = load_data(config)
	config = subpreprocess(config, "train")
	config = subpreprocess(config, "test")
	embedding, emb_dim = load_embeddings_file(config["embeddings"], lower=True)
	# Assert that emb_dim is equal to we_dim
	assert(emb_dim == config["we_dim"])
	config["w2i"] = words_to_indices(config["train"])
	config["w2i"], config["wembedding"] = construct_embedding(config, config["w2i"], embedding)
	config["vocab_size"] = len(config["wembedding"])
	return config

def encode_tags(config, tags):
	"""one-hot encode labels"""
	tag2id = reverse_map(config["tag_map"])
	tags = np.array([int(tag2id[tag]) for tag in tags])
	encoded_tags = (np.arange(len(tag2id)) == tags[:, None]).astype(np.float32)
	return encoded_tags

def doc_to_tensor(config, sess, graph, tokens, tags=None, train=False):
	"""Convert a document to a tensor representation of documents
	and labels"""

	w2i = config["w2i"]
	c2i = config["c2i"]
	max_tokens = config["max_tokens"]
	char_inputs, rev_char_inputs = [], []
	word_indices = [w2i.get(w, w2i["_UNK"]) for w in tokens]

	# Encode Tags
	if tags is not None:
		encoded_tags = encode_tags(config, tags)
	else:
		encoded_tags = None

	# Lookup Character Indices
	tokens = [["<w>"] + list(t) + ["</w>"] for t in tokens]
	max_t_len = len(max(tokens, key=len))

	for i in range(max_tokens):

		t = tokens[i] if i < len(tokens) else False

		if not t:
			char_inputs.append([0] * max_t_len)
			continue

		char_inputs.append([])

		for ii in range(max_t_len):
			char_index = c2i.get(t[ii], 0) if ii < len(t) else 0
			char_inputs[i].append(char_index)

	char_inputs = np.array(char_inputs)
	char_inputs = np.transpose(char_inputs, (1, 0))

	word_lengths = [len(tokens[i]) if i < len(tokens) else 0 for i in range(max_tokens)]

	return char_inputs, word_lengths, word_indices, encoded_tags

def char_encoding(config, graph, doc_len):
	"""Create graph nodes for character encoding"""

	c2i = config["c2i"]
	max_tokens = config["max_tokens"]

	with graph.as_default():

		# Character Embedding
		word_lengths = tf.placeholder(tf.int64, [None], name="word_lengths")
		word_lengths = tf.gather(word_lengths, tf.range(tf.to_int32(doc_len)))
		char_inputs = tf.placeholder(tf.int32, [None, max_tokens], name="char_inputs")
		cembed_matrix = tf.Variable(tf.random_uniform([len(c2i.keys()), config["ce_dim"]], -0.25, 0.25), name="cembeds")
		
		char_inputs = tf.transpose(char_inputs, perm=[1,0])
		cembeds = tf.nn.embedding_lookup(cembed_matrix, char_inputs, name="ce_lookup")
		cembeds = tf.gather(cembeds, tf.range(tf.to_int32(doc_len)), name="actual_ce_lookup")
		cembeds = tf.transpose(cembeds, perm=[1,0,2])

		# Create LSTM for Character Encoding
		fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"], state_is_tuple=True)
		bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"], state_is_tuple=True)

		# Encode Characters with LSTM
		options = {
			"dtype": tf.float32,
			"sequence_length": word_lengths,
			"time_major": True
		}

		(output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, cembeds, **options)
		output_fw = tf.transpose(output_fw, perm=[1,0,2])
		output_bw = tf.transpose(output_bw, perm=[1,0,2])

		return output_fw, output_bw, word_lengths

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	c2i = config["c2i"]
	max_tokens = config["max_tokens"]

	# Build Graph
	with graph.as_default():

		# Character Embedding
		doc_len = tf.placeholder(tf.int64, None, name="doc_length")
		train = tf.placeholder(tf.bool, name="train")
		tf.set_random_seed(config["seed"])
		last_state, rev_last_state, word_lengths = char_encoding(config, graph, doc_len)

		# Word Embedding
		word_inputs = tf.placeholder(tf.int32, [None], name="word_inputs")
		wembed_matrix = tf.Variable(tf.constant(0.0, shape=[config["vocab_size"], config["we_dim"]]), trainable=True, name="wembed_matrix")
		embedding_placeholder = tf.placeholder(tf.float32, [config["vocab_size"], config["we_dim"]], name="embedding_placeholder")
		assign_wembedding = tf.assign(wembed_matrix, embedding_placeholder, name="assign_wembedding")
		wembeds = tf.nn.embedding_lookup(wembed_matrix, word_inputs, name="we_lookup")
		wembeds = tf.identity(wembeds, name="actual_we_lookup")

		# Combine Embeddings
		char_embeds = last_relevant(last_state, word_lengths, "char_embeds")
		rev_char_embeds = last_relevant(rev_last_state, word_lengths, "rev_char_embeds")
		combined_embeddings = tf.concat(1, [wembeds, char_embeds, tf.reverse(rev_char_embeds, [True, False])], name="combined_embeddings")

		# Cells and Weights
		fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"], state_is_tuple=True)
		bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"], state_is_tuple=True)
		fw_network = tf.nn.rnn_cell.MultiRNNCell([fw_lstm]*config["num_layers"], state_is_tuple=True)
		bw_network = tf.nn.rnn_cell.MultiRNNCell([bw_lstm]*config["num_layers"], state_is_tuple=True)

		weight = tf.Variable(tf.random_uniform([config["h_dim"] * 2, len(config["tag_map"])]), name="weight")
		bias = tf.Variable(tf.random_uniform([len(config["tag_map"])]))

		def model(combined_embeddings, noise_sigma=0.0):
			"""Model to train"""

			combined_embeddings = tf.cond(train, lambda: tf.add(tf.random_normal(tf.shape(combined_embeddings)) * noise_sigma, combined_embeddings), lambda: combined_embeddings)
			batched_input = tf.expand_dims(combined_embeddings, 0)

			options = {
				"dtype": tf.float32,
				"sequence_length": tf.expand_dims(doc_len, 0)
			}

			(outputs_fw, outputs_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_network, bw_network, batched_input, **options)

			# Add Noise and Predict
			concat_layer = tf.concat(2, [outputs_fw, tf.reverse(outputs_bw, [False, True, False])], name="concat_layer")
			concat_layer = tf.cond(train, lambda: tf.add(tf.random_normal(tf.shape(concat_layer)) * noise_sigma, concat_layer), lambda: concat_layer)
			prediction = tf.log(tf.nn.softmax(tf.matmul(tf.squeeze(concat_layer, [0]), weight) + bias), name="model")
			return prediction

		network = model(combined_embeddings, noise_sigma=config["noise_sigma"])

		# Calculate Loss and Optimize
		labels = tf.placeholder(tf.float32, shape=[None, len(config["tag_map"].keys())], name="y")
		loss = tf.neg(tf.reduce_sum(network * labels), name="loss")
		optimizer = tf.train.GradientDescentOptimizer(config["learning_rate"]).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver, run_options, run_metadata):
	"""Train the model"""

	eras = config["eras"]
	dataset = config["dataset"]
	train = config["train"]
	train_index = list(range(len(train)))
	sess.run(get_op(graph, "assign_wembedding"), feed_dict={get_tensor(graph, "embedding_placeholder:0"): config["wembedding"]})

	# Train the Model
	for step in range(eras):
		count = 0
		random.shuffle(train_index)
		total_loss = 0
		total_tagged = 0

		logging.info("ERA: " + str(step))
		np.set_printoptions(threshold=np.inf)

		for t_index in train_index:

			count += 1

			tokens, tags = train[t_index]
			char_inputs, word_lengths, word_indices, labels = doc_to_tensor(config, sess, graph, tokens, tags=tags, train=True)

			feed_dict = {
				get_tensor(graph, "char_inputs:0") : char_inputs,
				get_tensor(graph, "word_inputs:0") : word_indices,
				get_tensor(graph, "word_lengths:0") : word_lengths,
				get_tensor(graph, "doc_length:0"): len(tokens),
				get_tensor(graph, "y:0"): labels,
				get_tensor(graph, "train:0"): True
			}

			# Collect GPU Profile
			if config["profile_session"]:
				optimizer_out, loss = sess.run([get_op(graph, "optimizer"), get_tensor(graph, "loss:0")], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('timeline.json', 'w') as f:
					f.write(ctf)
					sys.exit()

			# Run Training Step
			optimizer_out, loss = sess.run([get_op(graph, "optimizer"), get_tensor(graph, "loss:0")], feed_dict=feed_dict)
			total_loss += loss
			total_tagged += len(word_indices)

			# Log
			if count % 250 == 0:
				logging.info("count: " + str(count))
				logging.info("loss: " + str(total_loss/total_tagged))
				logging.info("%d" % (count / len(train_index) * 100) + "% complete with era " + str(step))

		# Evaluate Model
		test_accuracy = evaluate_testset(config, graph, sess, config["test"])

	final_model_path = "./meerkat/longtail/models/"
	os.makedirs(final_model_path, exist_ok=True)
	w2i_to_json(config["w2i"], final_model_path)
	save_models(saver, sess, final_model_path)
	return final_model_path

def save_models(saver, sess, path):
	ckpt_path = path + "bilstm.ckpt"
	meta_path = path + "bilstm.meta"
	model_path = saver.save(sess, ckpt_path)
	os.rename(ckpt_path+".meta", meta_path)
	logging.info("Save model to {0}, {1}".format(ckpt_path, meta_path))

def w2i_to_json(w2i, path):
	path = path + "w2i.json"
	with open(path, "w") as fp:
		json.dump(w2i, fp)
	logging.info("Save w2i to {0}".format(path))

def evaluate_testset(config, graph, sess, test):
	"""Check error on test set"""

	total_count = 0
	total_correct = 0
	count = 0
	logging.info("---ENTERING EVALUATION---")

	for i in range(len(test)):

		count += 1

		if count % 500 == 0:
			logging.info("%d" % (count / len(test) * 100) + "% complete with evaluation")

		tokens, tags = test[i]
		char_inputs, word_lengths, word_indices, labels = doc_to_tensor(config, sess, graph, tokens, tags=tags)
		total_count += len(tokens)

		feed_dict = {
			get_tensor(graph, "char_inputs:0") : char_inputs,
			get_tensor(graph, "word_inputs:0") : word_indices,
			get_tensor(graph, "word_lengths:0") : word_lengths,
			get_tensor(graph, "doc_length:0"): len(tokens),
			get_tensor(graph, "train:0"): False
		}

		output = sess.run(get_tensor(graph, "model:0"), feed_dict=feed_dict)

		correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels, 1))
		total_correct += correct_count

	test_accuracy = 100.0 * (total_correct / total_count)
	logging.info("Test accuracy: %.2f%%" % test_accuracy)
	logging.info("Correct count: " + str(total_correct))
	logging.info("Total count: " + str(total_count))

	return test_accuracy

def run_session(config, graph, saver):
	"""Run Session"""

	model_path = config["model_path"]

	devices = {'GPU': 0} if config["use_cpu"] else {}
	tf_config = tf.ConfigProto(device_count=devices)

	with tf.Session(graph=graph, config=tf_config) as sess:

		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()
		tf.initialize_all_variables().run()
		final_model_path = train_model(config, graph, sess, saver, run_options, run_metadata)

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	config = preprocess(config)
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
