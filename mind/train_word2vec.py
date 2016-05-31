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
wiki = LineSentence("data/output/wiki.en.text")

# Load Thoughts
prophet = pd.read_csv("data/input/thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
thoughts = [t.split() for t in list(prophet["Thought"])]

# Train
model = Word2Vec(size=600, window=5, min_count=25, workers=multiprocessing.cpu_count())

print("Beginning to build vocab")

model.build_vocab(wiki)
model.train(wiki)
model.train(thoughts)

# Evaluate
print(model.most_similar("prophet"))
print(model.most_similar("cognicism"))

# Save
model.save("models/prophet_word2vec.bin")
