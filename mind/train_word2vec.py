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
import argparse
import operator
import logging
import multiprocessing

import pandas as pd

from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def parse_arguments(args):
	""" Create the parser """

	parser = argparse.ArgumentParser(description="Process wikipedia data set and train word2vec model")
	parser.add_argument('--wiki', '-wiki', required=False, default="data/enwiki-latest-pages-articles.xml.bz2", help='Path to wiki source for processing')
	parser.add_argument('--out', '-out', required=False, default="data/wiki.en.text", help='Output file name for processed wiki data')
	
	return parser.parse_args(args)

def make_corpus(in_f, out_f):
	"""Convert Wikipedia xml dump file to text corpus"""

	output = open(out_f, 'w')
	wiki = WikiCorpus(in_f)

	i = 0

	for text in wiki.get_texts():
		output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
		i = i + 1
		if (i % 10000 == 0):
			print('Processed ' + str(i) + ' articles')

	output.close()
	print('Processing complete!')

def main():
	"""Run module from command line"""

	# Load Wiki Data
	wiki = LineSentence("/data/home/ec2-user/data/wiki.en.text")
	# wiki = LineSentence("data/wiki_02.txt")

	# Load Thoughts
	prophet = pd.read_csv("data/thoughts.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)
	thoughts = [re.sub('[^A-Za-z0-9]+', ' ', t).lower().split() for t in list(prophet["Thought"])]

	# Create Unique Tokens for Key Words
	for thought in thoughts:
		for i, token in enumerate(thought):
			if token == "prophet":
				thought[i] = "matt_prophet"

	# Create Model
	model = Word2Vec(size=600, window=5, min_count=20, workers=multiprocessing.cpu_count())

	# Build Vocab
	print("Beginning to build vocab")
	model.build_vocab(wiki)
	model.build_vocab(thoughts, update=True)

	if "confluesce" in model.wv:
		print("Confluesce found")

	# Train
	print("train on wiki")
	model.train(wiki, total_examples=model.corpus_count, epochs=model.epochs)
	print("train on prophet")
	model.train(thoughts, total_examples=model.corpus_count, epochs=model.epochs)

	print("repeat train on prophet")
	for i in range(0, 25):
		model.train(thoughts, total_examples=model.corpus_count, epochs=model.epochs)

	# Evaluate
	print("most similar to matt_prophet")
	print(model.wv.most_similar("matt_prophet"))
	print("most similar to prophet")
	print(model.wv.most_similar("prophet"))

	# Save
	model.save("models/special_prophet_word2vec.bin")

if __name__ == "__main__":

	args = parse_arguments(sys.argv[1:])
	# make_corpus(args.wiki, args.out)
	main()
