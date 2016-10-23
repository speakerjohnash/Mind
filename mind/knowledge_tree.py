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
	thoughts = list(seers["prophet"]["Thought"]) + list(seers["msevrens@yodlee.com"]["Thought"]) + list(seers["msevrens"]["Thought"])
	ken = vectorize(thoughts, min_df=1)
	sorted_ken = sorted(ken.items(), key=operator.itemgetter(1))
	sorted_ken.reverse()

	keys_in_word2vec = model.vocab.keys()
	tokens = [x[0] for x in sorted_ken[0:5000] if x[0] in keys_in_word2vec]

def knowledge_tree():
	"""Build JSON for Tree of Knowledge"""
	
	# Load Model
	print("Loading model...")
	model = gensim.models.Word2Vec.load('models/prophet_word2vec')  # C binary format
	print("Loading model: Done")

	# Load Top Words
	df = pd.read_csv("data/input/thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
	seers_grouped = df.groupby('Seer', as_index=False)
	seers = dict(list(seers_grouped))
	thoughts = list(seers["prophet"]["Thought"]) + list(seers["msevrens@yodlee.com"]["Thought"]) + list(seers["msevrens"]["Thought"])
	ken = vectorize(thoughts, min_df=1)
	sorted_ken = sorted(ken.items(), key=operator.itemgetter(1))
	sorted_ken.reverse()

	keys_in_word2vec = model.vocab.keys()
	tokens = [x[0] for x in sorted_ken[0:5000] if x[0] in keys_in_word2vec]

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

	dict_2_json(similarity_lookup, "fruiting" + "_word2vec_tree.json")

	return similarity_lookup

def tree_cloud(similarity_lookup, depth=3, seed_word="word2vec"):
	"""Build CSV for Tree Cloud"""

	weights = defaultdict(list)
	used_words = [seed_word]
	max_depth = depth
	num_children = 5

	if seed_word not in similarity_lookup:
		print("Seed word not found, please try another word")
		sys.exit()

	def grow_branch(word, level):

		num_twigs = num_children;

		if level <= 0 or word not in similarity_lookup:
			return

		if len(similarity_lookup[word]) < num_children:
			num_twigs = len(similarity_lookup[word])

		for i in range(num_twigs):

			cur_word = similarity_lookup[word][i]["w"]

			if cur_word in used_words:
				continue

			used_words.append(cur_word)
			weights[level].append(cur_word)

		for word in weights[level]:
			new_level = level - 2
			grow_branch(word, new_level)

	weights[max_depth + 5].append(seed_word)
	grow_branch(seed_word, max_depth)

	dict_list = []

	for weight, words in weights.items():
		for word in words:
			dict_list.append({
				"weight": weight,
				"word": word
			})

	with open('data/output/tree_cloud.tab', 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, ["weight", "word"], delimiter="\t")
		dict_writer.writeheader()
		dict_writer.writerows(dict_list)

if __name__ == "__main__":
	#ken = load_ken()
	#similarity_lookup = knowledge_tree(ken)

	similarity_lookup = load_json("data/output/word2vec_tree.json")
	tree_cloud(similarity_lookup, depth=10, seed_word="garden")
