#!/usr/local/bin/python3

"""This module generates a json lookup 
for the tree of knowledge

@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.knowledge_tree [user]
# python3 -m mind.knowledge_tree msevrens

#####################################################

import os
import sys
import re
import math
import json
import csv
import operator

import editdistance
import pandas as pd
from collections import defaultdict

import gensim
from gensim.models import word2vec

from mind.tools import dict_2_json, vectorize, word_count, load_json

def load_ken():
	"""Load top words in Prophet data"""

	df = pd.read_csv("data/input/thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
	seers_grouped = df.groupby('Seer', as_index=False)
	seers = dict(list(seers_grouped))
	thoughts = list(seers[sys.argv[1]]["Thought"])
	ken = vectorize(thoughts, min_df=1)
	sorted_ken = sorted(ken.items(), key=operator.itemgetter(1))
	sorted_ken.reverse()

	return sorted_ken

def knowledge_tree(ken):
	"""Build JSON for Tree of Knowledge"""
	
	# Load Model
	print("Loading model...")
	model = gensim.models.Word2Vec.load('models/prophet_word2vec')  # C binary format
	print("Loading model: Done")

	keys_in_word2vec = model.vocab.keys()
	tokens = [x[0] for x in ken[0:5000] if x[0] in keys_in_word2vec]
	number_words = len(model.vocab)
	vocab = list(model.vocab.keys())

	similarity_lookup = {}

	for i in range(0, number_words):
		word = vocab[i]

		if word not in tokens or word.isdigit():
			continue

		nearest_words = model.most_similar(positive=[word], negative=[], topn=5000)
		nearest_tokens = []
		non_lexicon = []

		for w in nearest_words:
			if w[0] in tokens:
				nearest_tokens.append([w[0], w[1], True])
			else:
				non_lexicon.append([w[0], w[1], False])

		nearest_words = nearest_tokens[0:19]

		if len(non_lexicon) == 0 or len(nearest_words) == 0:
			continue

		if non_lexicon[0][1] > nearest_tokens[0][1]:
			if editdistance.eval(word, non_lexicon[0][0]) > 2:
				print("Word: " + word)
				print("Nearest out of lexicon word: " + non_lexicon[0][0])
				nearest_words.append(non_lexicon[0])

		number_nearest_words = len(nearest_words)

		similarity_lookup[word] = [{"w": nw[0], "d": round(nw[1], 3), "l": nw[2]} for nw in nearest_words]

	print(similarity_lookup)

	dict_2_json(similarity_lookup, "fruiting" + "_leah_tree.json")

	return similarity_lookup

if __name__ == "__main__":
	ken = load_ken()
	similarity_lookup = knowledge_tree(ken)
