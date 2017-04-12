#!/usr/local/bin/python3

"""This module demonstrates the scoring model
behind Prophet

Created on Jan 09, 2017
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.train_mind config/mind_model_config.json
# python3 -m mind.train_mind [config_file]

# http://www.metaculus.com/help/scoring
# https://arxiv.org/pdf/1610.10099.pdf
# https://github.com/paarthneekhara/byteNet-tensorflow/

#####################################################

import logging
import math
import sys
import argparse

import numpy as np
import tensorflow as tf

from mind.mind_models import TruthModel
from mind.data_loaders import PretrainData, TranslationData
from mind.tools import load_dict_list, load_json, load_piped_dataframe

# Utility
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="*")
parser.add_argument(
	'--resume_model', 
	type=str, 
	default=None, 
	help='Pre-trained model path, to resume from'
)

def train_translator(config):
	"""Train a translator"""

	epochs = config["options"]["max_epochs"]

	# Load Data
	paired_sentences = TranslationData(config["options"]["bucket_quant"], config)
	buckets, source_vocab, target_vocab, frequent_keys = paired_sentences.bucket_data()

	# Configure Model Options
	model_options = config["translator"]
	model_options["n_source_quant"] = len(source_vocab)
	model_options["n_target_quant"] = len(target_vocab)
	model_options["sample_size"] = 10
	model_options["source_mask_chars"] = [source_vocab["padding"]]
	model_options["target_mask_chars"] = [target_vocab["padding"]]

	last_saved_model_path = None

	if "resume_model" in config:
		last_saved_model_path = config["resume_model"]

	# Train Model
	for i in range(1, epochs):

		cnt = 0

		for _, key in frequent_keys:

			key = int(key)
			cnt += 1

			if key not in buckets:
				continue
			
			print(("Key", cnt, key))
			if key > 400:
				continue
			
			if len(buckets[key]) < model_options["batch_size"]:
				print(("Bucket too small", key))
				continue

			sess = tf.InteractiveSession()

			batch_no = 0
			batch_size = model_options["batch_size"]

			# Build Model
			model = TruthModel(model_options)
			tensors = model.build_translation_model(sample_size=key)
			
			# Build Optimizer
			lr = config["options"]["learning_rate"]
			beta1 = config["options"]["adam_momentum"]
			adam = tf.train.AdamOptimizer(lr, beta1=beta1)
			optim = adam.minimize(tensors["loss"], var_list=tensors["variables"])

			# Initialize Variables and Summary Writer
			train_writer = tf.summary.FileWriter('logs/', sess.graph)
			tf.global_variables_initializer().run()

			saver = tf.train.Saver()

			# Restore previous checkpoint if existing
			if last_saved_model_path:
				saver.restore(sess, last_saved_model_path)

			# Training Step
			while (batch_no + 1) * batch_size < len(buckets[key]):

				source, target = paired_sentences.load_batch( 
					buckets[key][batch_no * batch_size : (batch_no + 1) * batch_size] 
				)

				tensors_to_get = [
					optim, 
					tensors['loss'], 
					tensors['prediction'], 
					tensors['merged_summary']
				]

				feed_dict = {
					tensors['source_sentence'] : source,
					tensors['target_sentence'] : target
				}

				# Run Session and Expand Outputs
				outputs = sess.run(tensors_to_get, feed_dict=feed_dict)
				_, loss, prediction, summary = outputs

				# Write to Summary
				train_writer.add_summary(summary, batch_no * (cnt + 1))
				print(("Loss", loss, batch_no, len(buckets[key])/batch_size, i, cnt, key))
				
				# Print Results to Terminal
				print("******")
				print(("Source ", paired_sentences.char_indices_to_string(source[0], source_vocab)))
				print("---------")
				print(("Target ", paired_sentences.word_indices_to_string(target[0], target_vocab)))
				print("----------")
				print(("Prediction ", paired_sentences.word_indices_to_string(prediction[0:int(key)], target_vocab)))
				print("******")

				batch_no += 1

				if batch_no % 5000 == 0:
					save_path = saver.save(sess, "models/model_translation_epoch_{}_{}.ckpt".format(i, cnt))
					last_saved_model_path = "models/model_translation_epoch_{}_{}.ckpt".format(i, cnt)

			# Save Checkpoint
			save_path = saver.save(sess, "models/model_translation_epoch_{}.ckpt".format(i))
			last_saved_model_path = "models/model_translation_epoch_{}.ckpt".format(i)

			tf.reset_default_graph()
			sess.close()

def pretrain_prophet(config):
	"""Train a language model via sequential thought prediction"""

	epochs = config["options"]["max_epochs"]

	# Load Data
	thought_stream = PretrainData(config["options"]["bucket_quant"], config)
	buckets, source_vocab, target_vocab, frequent_keys = thought_stream.bucket_data()

	# Configure Model Options
	model_options = config["prophet"]
	model_options["n_source_quant"] = len(source_vocab)
	model_options["n_target_quant"] = len(target_vocab)
	model_options["source_mask_chars"] = [source_vocab["padding"]]
	model_options["target_mask_chars"] = [target_vocab["padding"]]

	last_saved_model_path = None

	if "resume_model" in config:
		last_saved_model_path = config["resume_model"]

	# Train Model
	for i in range(1, epochs):

		cnt = 0

		key = model_options["sample_size"]
		cnt += 1
		
		print(("Key", cnt, key))

		sess = tf.InteractiveSession()

		batch_no = 0
		batch_size = model_options["batch_size"]

		# Build Model
		model = TruthModel(model_options)
		tensors = model.build_truth_model(sample_size=key)
		
		# Build Optimizer
		lr = config["options"]["learning_rate"]
		beta1 = config["options"]["adam_momentum"]
		adam = tf.train.AdamOptimizer(lr, beta1=beta1)
		optim = adam.minimize(tensors["loss"], var_list=tensors["variables"])

		# Initialize Variables and Summary Writer
		train_writer = tf.summary.FileWriter('logs/', sess.graph)
		tf.global_variables_initializer().run()

		saver = tf.train.Saver()

		# Restore previous checkpoint if existing
		if last_saved_model_path:
			saver.restore(sess, last_saved_model_path)

		# Training Step
		while (batch_no + 1) < len(buckets[key]):

			# TODO: Training Step

			if batch_no % 5000 == 0:
				save_path = saver.save(sess, "models/model_pretrain_epoch_{}_{}.ckpt".format(i, cnt))
				last_saved_model_path = "models/model_pretrain_epoch_{}_{}.ckpt".format(i, cnt)

		# Save Checkpoint
		save_path = saver.save(sess, "models/model_pretrain_epoch_{}.ckpt".format(i))
		last_saved_model_path = "models/model_pretrain_epoch_{}.ckpt".format(i)

		tf.reset_default_graph()
		sess.close()

def train_prophet(config):
	"""Train a truth model"""

	epochs = config["options"]["max_epochs"]
	model_options = config["prophet"]

def main():
	"""Run module from command line"""

	args = parser.parse_args()
	config = load_json(args.config[0])
	model_type = config["options"]["model_type"]

	if args.resume_model:
		config["resume_model"] = args.resume_model

	if model_type == "translator":
		train_translator(config)
	elif model_type == "predictor":
		pretrain_prophet(config)
	elif model_type == "prophet":
		train_prophet(config)

if __name__ == "__main__":
	main()