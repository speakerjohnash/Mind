#!/usr/local/bin/python3.3

"""A collection of functions to be called by multiple modules throughout 
Prophet Mind

Created on Jan 21, 2015
@author: Matthew Sevrens
"""

import csv
import json
import os
import sys

import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

def load_dict_list(file_name, encoding='utf-8', delimiter=","):
	"""Loads a dictionary of input from a file into a list."""

	with open(file_name, 'r', encoding="utf-8", errors='replace') as input_file:
		dict_list = list(csv.DictReader(input_file, delimiter=","))
		
	return dict_list

def to_stdout(string, errors="replace"):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open(filename, 'w') as fp:
		json.dump(obj, fp, indent=4)

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""
	print(*(to_stdout(str(o), errors) for o in objs))

def safe_input(prompt=""):
	"""Safely input a string"""

	try:
		result = input(prompt)
		return result
	except KeyboardInterrupt:
		sys.exit()
	except:
		return ""

def progress(i, my_list, message=""):
	"""Display progress percent in a loop"""
	my_progress = (i / len(my_list)) * 100
	my_progress = str(round(my_progress, 1)) + "% " + message
	sys.stdout.write('\r')
	sys.stdout.write(my_progress)
	sys.stdout.flush()

def safely_remove_file(filename):
	"""Safely removes a file"""
	print("Removing {0}".format(filename))
	try:
		os.remove(filename)
	except OSError:
		print("Unable to remove {0}".format(filename))
	print("File removed.")

def reverse_map(label_map, key='label'):
	"""reverse {key : {category, label}} to {label: key} and
	{key: value} to {value: key} dictionary}"""
	get_key = lambda x: x[key] if isinstance(x, dict) else x
	reversed_label_map = dict(zip(map(get_key, label_map.values()),
		label_map.keys()))
	return reversed_label_map

def load_json(filename_or_dict):
	"""Load a json file provided a filename"""
	if isinstance(filename_or_dict, str):
		input_file = open(filename_or_dict, encoding='utf-8')
		json_dict = json.loads(input_file.read())
		input_file.close()
		return json_dict
	return filename_or_dict

def load_piped_dataframe(filename, chunksize=False, usecols=False):
	"""Load piped dataframe from file name"""

	options = {
		"quoting": csv.QUOTE_NONE,
		"na_filter": False,
		"encoding": "utf-8",
		"sep": "|",
		"error_bad_lines": False
	}

	if usecols:
		columns = usecols
		options["usecols"] = usecols
	else:
		with open(filename, 'r') as reader:
			header = reader.readline()
		columns = header.split("|")

	options["dtype"] = {c: "object" for c in columns}

	if isinstance(chunksize, int):
		options["chunksize"] = chunksize

	return pd.read_csv(filename, **options)
	
def get_write_func(filename, header):
	
	file_exists = False
	
	def write_func(data):
		if len(data) > 0:
			nonlocal file_exists
			mode = "a" if file_exists else "w"
			add_head = False if file_exists else header
			df = pd.DataFrame(data)
			df.to_csv(filename, mode=mode, index=False, header=add_head)
			file_exists = True
		else:
			open(filename, 'a').close()
	
	return write_func

def vectorize(corpus, min_df=1):
	"""Vectorize text corpus"""

	tknzr = TweetTokenizer().tokenize

	def tokenizer(thought):
		output = tknzr(thought)
		output = [o for o in output if len(o) > 1]
		return output

	vectorizer = CountVectorizer(min_df=min_df, tokenizer=tokenizer, ngram_range=(1, 1), stop_words='english')
	count_vector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = count_vector.shape
	vocab = vectorizer.get_feature_names()

	wc = word_count(vocab, count_vector, num_samples)

	return wc

def word_count(vocab, count_vector, num_samples):
	"""Count words"""

	np.clip(count_vector, 0, 1, out=count_vector)
	dist = np.sum(count_vector, axis=0)
	dist = dist.tolist()

	wc = dict(zip(vocab, dist))

	return wc

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

if __name__ == "__main__":
	print("This module is a library that contains useful functions; it should not be run from the console.")
