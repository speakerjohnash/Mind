#!/usr/local/bin/python3.4

"""This module defines the Prophet Mind Wordstream Analyzer web service API

Created on Sep 04, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# curl -X POST -d @schemas/wordstream/example_input.json http://localhost:443/wordstream/ | python3.3 -m json.tool

#####################################################

import json

import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from tornado_json.requesthandlers import APIHandler
from mind.api import schema

def vectorize(corpus, min_df=1):
	"""Vectorize text corpus"""

	tknzr = TweetTokenizer().tokenize

	def tokenizer(thought):
		output = tknzr(thought)
		output = [o for o in output if len(o) > 1]
		return output

	vectorizer = CountVectorizer(min_df=min_df, tokenizer=tokenizer, ngram_range=(1,1), stop_words='english')
	count_vector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = count_vector.shape
	vocab = vectorizer.get_feature_names()
	wc = word_count(vocab, count_vector, num_samples)

	return wc

def word_count(vocab, count_vector, num_samples):
	"""Count words"""

	np.clip(count_vector, 0, 1, out=count_vector)
	dist = np.sum(count_vector, axis=0)
	dist = dist.tolist()

	wc = dict(zip(vocab, dist))

	return wc

def process_by_day(data):
	"""Process data by data and return in proper format"""

	output = {"days" : []}

	for day in data["days"]:

		today = {"word_list": [], "date": day["date"]}

		try:
			counted = vectorize(day["thoughts"])
		except:
			continue;

		for word, count in counted.items():
			today["word_list"].append({"word": word, "count": count})

		output["days"].append(today)

	return output

class Wordstream_Analysis(APIHandler):
	"""This class handles Wordstream Analysis for Prophet"""

	with open("schemas/wordstream/schema_input.json") as data_file:
		schema_input = json.load(data_file)

	with open("schemas/wordstream/example_input.json") as data_file:
		example_input = json.load(data_file)

	with open("schemas/wordstream/schema_output.json") as data_file:
		schema_output = json.load(data_file)

	with open("schemas/wordstream/example_output.json") as data_file:
		example_output = json.load(data_file)

	@schema.validate(
		input_schema=schema_input,
		input_example=example_input,
		output_schema=schema_output,
		output_example=example_output
	)

	def post(self):
		"""Handle post requests"""

		data = json.loads(self.request.body.decode())
		results = process_by_day(data)
		return results

	def get(self):
		"""Handle get requests"""
		return None

if __name__ == "__main__":
	print("This module is a Class; it should not be run from the console.")