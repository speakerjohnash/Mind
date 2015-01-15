#!/usr/local/bin/python3.3

"""This module clusters word2vec word vectors
for exploration of datasets

Created on Dec 4, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 cluster_words.py [word2vec_model]

#####################################################

import csv
import sys
import operator

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans as kmeans

def to_stdout(string, errors='replace'):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""

	print(*(to_stdout(str(o), errors) for o in objs))

def cluster_vectors(word2vec):
	"""Clusters a set of word vectors"""

	n_clusters = int(word2vec.syn0.shape[0] / 20)
	clusters = kmeans(n_clusters=500, max_iter=100, batch_size=200, n_init=1, init_size=2000)
	tokens = word2vec.vocab.keys()

	X = np.array([word2vec[t].T for t in tokens])

	clusters.fit(X)
	word_clusters = {word:label for word, label in zip(tokens, clusters.labels_)}
	sorted_clusters = sorted(word_clusters.items(), key=operator.itemgetter(1))
	#safe_print(clusters.cluster_centers_)

	safe_print(sorted_clusters)

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""

	word2vec = Word2Vec.load(sys.argv[1])
	cluster_vectors(word2vec)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
