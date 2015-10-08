#!/usr/local/bin/python3.4

"""This module defines the Prophet Mind Wordstream Analyzer web service API

Created on Sep 04, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# curl -X POST -d @schemas/wordstream/example_input.json http://localhost:443/wordstream/

#####################################################

import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from tornado_json.requesthandlers import APIHandler
from tornado_json import schema

def vectorize(corpus, min_df=1):
	"""Vectorize text corpus"""

	vectorizer = CountVectorizer(min_df=min_df, ngram_range=(1,1), stop_words='english')
	countVector = vectorizer.fit_transform(corpus).toarray()
	num_samples, num_features = countVector.shape
	vocab = vectorizer.get_feature_names()
	word_count = wordCount(vocab, countVector, num_samples)

	return word_count

def word_count(vocab, countVector, num_samples):
	"""Count words"""

	np.clip(countVector, 0, 1, out=countVector)
	dist = np.sum(countVector, axis=0)
	dist = dist.tolist()

	word_count = dict(zip(vocab, dist))

	return word_count

def process_by_day(data):
	"""Process data by data and return in proper format"""

	for day in data["days"]:
		print(day["thoughts"])

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