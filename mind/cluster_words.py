#!/usr/local/bin/python3.4

"""This module clusters word2vec word vectors
for exploration of datasets

Created on Dec 4, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.cluster_words [word2vec_model]
# Word2Vec available at: https://code.google.com/p/word2vec/

#####################################################

import csv
import sys
import operator
import collections
import pickle

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans as kmeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from mind.tools import dict_2_json

def to_stdout(string, errors='replace'):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""

	print(*(to_stdout(str(o), errors) for o in objs))

def save_token_subset(word2vec, word_list):
	"""Save a subset of token and associated vectors
	to a file for faster loading"""

	vector_dict = collections.defaultdict()

	for word in word_list:
		vector_dict[word] = word2vec[word]

	pickle.dump(vector_dict, open("models/prophet_vectors.pkl", "w"))

def vectorize(corpus, min_df=1):
	"""Vectorize text corpus"""

	vectorizer = CountVectorizer(min_df=min_df, ngram_range=(1,1), stop_words='english')
	countVector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()

	word_count = wordCount(vocab, countVector, num_samples)

	return word_count

def wordCount(vocab, countVector, num_samples):
	"""Count words"""

	np.clip(countVector, 0, 1, out=countVector)
	dist = np.sum(countVector, axis=0)
	dist = dist.tolist()

	word_count = dict(zip(vocab, dist))

	return word_count

def cluster_vectors(word2vec):
	"""Clusters a set of word vectors"""

	# Load Data
	df = pd.read_csv("data/input/thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
	seers_grouped = df.groupby('Seer', as_index=False)
	seers = dict(list(seers_grouped))
	thoughts = list(seers["msevrens"]["Thought"])
	ken = vectorize(thoughts, min_df=1)
	sorted_ken = sorted(ken.items(), key=operator.itemgetter(1))
	sorted_ken.reverse()

	n_clusters = int(word2vec.syn0.shape[0] / 20)
	keys_in_word2vec = word2vec.vocab.keys()
	tokens = [x[0] for x in sorted_ken[0:250] if x[0] in keys_in_word2vec]
	X = np.array([word2vec[t].T for t in tokens])

	# Create vector cache for faster load
	#save_token_subset(word2vec, tokens)
	
	# Clustering
	clusters = kmeans(n_clusters=75, max_iter=100, batch_size=200, n_init=10, init_size=30)
	clusters.fit(X)
	word_clusters = {word:label for word, label in zip(tokens, clusters.labels_)}
	sorted_clusters = sorted(word_clusters.items(), key=operator.itemgetter(1))
	collected = collections.defaultdict(list)

	for k in sorted_clusters:
		collected[k[1]].append(k[0])

	for key in collected.keys():
		safe_print(key, collected[key], "\n")

	# Visualize with t-SNE
	t_SNE(word2vec, tokens)

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	"""Plot labels using t-sne"""

	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))

	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig("data/output/" + filename)

def t_SNE(word2vec, tokens):
	"""Applies t-SNE to word vectors"""

	# Visualize an easy dataset for exploration
	top_embeddings = np.array([word2vec[t].T for t in tokens]).astype(np.float)

	# Dimensionality Reduction
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	low_dim_embs = tsne.fit_transform(top_embeddings)
	plot_with_labels(low_dim_embs, tokens)

	# Save Low Dimension Embeddings
	reduced = dict(zip(tokens, list(top_embeddings)))
	dict_2_json(reduced, "t-SNE_top_words_Prophet.json")

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""

	word2vec = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
	cluster_vectors(word2vec)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
