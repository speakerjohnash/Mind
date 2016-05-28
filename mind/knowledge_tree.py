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
import operator

import pandas as pd

from gensim.models import word2vec

from mind.tools import dict_2_json, vectorize, word_count

# Load Model
model_path = "models/word2vec.bin"
print("Loading model...")
model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)  # C binary format
print("Loading model: Done")

# Load Top Words
df = pd.read_csv("data/input/thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
seers_grouped = df.groupby('Seer', as_index=False)
seers = dict(list(seers_grouped))
thoughts = list(seers[sys.argv[1]]["Thought"])
ken = vectorize(thoughts, min_df=1)
sorted_ken = sorted(ken.items(), key=operator.itemgetter(1))
sorted_ken.reverse()

keys_in_word2vec = model.vocab.keys()
tokens = [x[0] for x in sorted_ken[0:2500] if x[0] in keys_in_word2vec]

number_words = len(model.vocab)
vocab = list(model.vocab.keys())

similarity_lookup = {}

for i in range(0, number_words):
	word = vocab[i]

	if word not in tokens:
		continue

	nearest_words = model.most_similar(positive=[word], negative=[], topn=5000)
	nearest_tokens = []

	for w in nearest_words:
		if w[0] in tokens:
			nearest_tokens.append(w)

	nearest_words = nearest_tokens[0:20]

	number_nearest_words = len(nearest_words)

	similarity_lookup[word] = [{"w": nw[0], "d": round(nw[1], 3)} for nw in nearest_words]

print(similarity_lookup)
dict_2_json(similarity_lookup, "patch615_word2vec_tree.json")

print("Finished!")
