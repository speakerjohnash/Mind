import os
import sys
import re
import math
import csv
import random
import collections

import numpy

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def to_stdout(string, errors='replace'):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""

	print(*(to_stdout(str(o), errors) for o in objs))

def load_dict_list(file_name):
	"""Loads a dictoinary of input from a file into a list"""

	with open(file_name, 'r', encoding="utf-8", errors='replace') as input_file:
		dict_list = list(csv.DictReader(input_file, delimiter=","))

	return dict_list

def progress(i, list, message=""):
	"""Display progress percent in a loop"""

	progress = (i / len(list)) * 100
	progress = str(round(progress, 1)) + "% " + message
	sys.stdout.write('\r')
	sys.stdout.write(progress)
	sys.stdout.flush()

def vectorize(corpus):
	"""Vectorize text corpus"""

	vectorizer = CountVectorizer(min_df=5, ngram_range=(1,1), stop_words='english')
	countVector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()

	termWeighting(vocab, countVector, corpus)
	distribution_dict = wordCount(vocab, countVector, num_samples)

	return distribution_dict

def wordCount(vocab, countVector, num_samples):
	"""Count words"""

	numpy.clip(countVector, 0, 1, out=countVector)
	dist = numpy.sum(countVector, axis=0)
	dist = dist.tolist()

	safe_print("Word Count")
	safe_print(dict(zip(vocab, dist)), "\n")

	distribution_dict = wordFrequency(vocab, dist, num_samples)

	return distribution_dict

def wordFrequency(vocab, dist, num_samples):

	dist[:] = [x / num_samples for x in dist]
	dist = numpy.around(dist, decimals=5).tolist()
	distribution_dict = dict(zip(vocab, dist))

	safe_print("Word Frequency")
	safe_print(distribution_dict, "\n")

	return distribution_dict

def termWeighting(vocab, countVector, corpus):
	"""Gives intution to word importance"""

	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(countVector)

	safe_print("Weights Per Word:")
	safe_print(dict(zip(vocab, numpy.around(transformer.idf_, decimals=5).tolist())), "\n")

def compareMinds(data_a, data_b):
	"""Compares available words"""

	deltas = {}

	for key in data_a:
		a_freq = data_a[key]
		b_freq = data_b.get(key, 0)
		deltas[key] = a_freq - b_freq

	safe_print(collections.OrderedDict(sorted(deltas.items(), key=lambda t: t[1])))

def collectThoughts(thoughts):
	"""Bucket data by user for easy access"""

	thinkers = collections.defaultdict(list)

	# Split into user buckets
	for thought in thoughts:
		thinker = thought.get("Seer", "")
		thinkers[thinker].append(thought)

	return thinkers

# TODO: 
# 1) import from drupal file DONE
# 2) Sort by user DONE
# 3) run countVectorizer on each day
# 4) Output and connect to thought stream code

def run_from_command():             
	"""Run if file invoked from command line"""
	
	params = {}
	thoughts = load_dict_list("data/input/Thoughts_August_17.csv")
	thinkers = collectThoughts(thoughts)

	pat_thoughts = thinkers['patch615']
	matt_thoughts = thinkers['msevrens']

	matt_thoughts = [thought['Thought'] for thought in matt_thoughts]
	matts_mind = vectorize(matt_thoughts)

if __name__ == "__main__":
	run_from_command()
