#!/usr/local/bin/python3

"""This module trains the Word2Vec
model further using Prophet data

@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.train_word2vec

#####################################################

import os
import sys
import re
import math
import json
import operator
import multiprocessing

import pandas as pd

from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence

# Load Wiki Data
# wiki = LineSentence("data/output/wiki.en.text")
wiki = LineSentence("data/wiki_02.txt")

# Load Thoughts
prophet = pd.read_csv("data/thoughts.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
thoughts = [re.sub('[^A-Za-z0-9]+', ' ', t).lower().split() for t in list(prophet["Thought"])]

# Train
model = Word2Vec(size=600, window=5, min_count=20, workers=multiprocessing.cpu_count())

print("Beginning to build vocab")
model.build_vocab(wiki)

if "confluesce" in model.wv:
	print("Confluesce found")

model.train(wiki, total_examples=model.corpus_count, epochs=model.epochs)
model.train(thoughts, total_examples=model.corpus_count, epochs=model.epochs)

# Evaluate
print(model.wv.most_similar("prophet"))
# print(model.wv.most_similar("cognicism"))

# Save
model.save("models/prophet_word2vec.bin")
