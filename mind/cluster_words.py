#!/usr/local/bin/python3.4

"""This module clusters word2vec word vectors
for exploration of datasets

Created on Dec 4, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.4 -m mind.cluster_words [word2vec_model]
# Word2Vec available at: https://code.google.com/p/word2vec/

#####################################################

import csv
import sys
import operator
import collections
import pickle

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans as kmeans
from sklearn.manifold import TSNE

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

	pickle.dump(vector_dict, open("models/prophet_vectors.pkl", "wb" ))

def cluster_vectors(word2vec):
	"""Clusters a set of word vectors"""

	n_clusters = int(word2vec.syn0.shape[0] / 20)
	tokens = word2vec.vocab.keys()
	tokens = ["prophet", "thought", "mind", "truth", "thoughts", "time", "good", "meaning", "like", "future", "make", "context", "words", "self", "consciousness", "stream", "need", "just", "knowledge", "new", "today", "way", "does", "know", "people", "reality", "day", "idea", "life", "ideas", "use", "predictions", "conscious", "light", "word", "state", "things", "reflect", "information", "brain", "learning", "prediction", "better", "think", "past", "speak", "work", "right", "flow", "change", "learn", "meds", "http", "reflection", "feel", "mimi", "data", "true", "want", "human", "language", "best", "com", "truths", "real", "love", "means", "minds", "tomorrow", "world", "important", "tree", "users", "come", "really", "point", "search", "great", "site", "different", "predict", "private", "live", "tool", "path", "perception", "form", "concept", "meditation", "create", "able", "body", "seek", "needs", "used", "design", "flood", "place", "ken", "little", "waves", "subjective", "thinking", "thing", "sync", "set", "lot", "objective", "having", "rate", "present", "health", "using", "makes", "understand", "wisdom", "humans", "sunrise", "mental", "perceive", "ask", "far", "healthy", "current", "wave", "years", "religion", "feature", "long", "predictive", "energy", "post", "understanding", "build", "log", "permanence", "focus", "say", "cognitive", "contexts", "god", "lucid", "user", "lost", "machine", "year", "knowing", "neural", "useful", "questions", "end", "physical", "potential", "did", "old", "let", "clock", "control", "write", "rating", "link", "sense", "andy", "sun", "line", "button", "dream", "public", "usage", "challenging", "distributed", "book", "based", "going", "remember", "streams", "awareness", "root", "bound", "start", "share", "define", "semantic", "explore", "features", "statements", "question", "helps", "repeat", "reflections", "stretch"]
	X = np.array([word2vec[t].T for t in tokens])

	# Create vector cache for faster load
	save_token_subset(word2vec, tokens)
	
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

def t_SNE(word2vec, tokens):
	"""Applies t-SNE to word vectors"""

	# Visualize an easy dataset for exploration
	X = np.array([word2vec[t].T for t in tokens])

	# Dimensionality Reduction
	model = TSNE(n_components=2, random_state=0)
	X = model.fit_transform(X.astype(np.float))
	reduced = dict(zip(tokens, list(X)))
	dict_2_json(reduced, "t-SNE_top_words_Prophet.json")

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""

	word2vec = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
	cluster_vectors(word2vec)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
