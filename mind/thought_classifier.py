#!/usr/local/bin/python3.3

"""This module generates and trains classificaton models for
using SciKit Learn. Specifically it uses a Stochastic 
Gradient Descent classifier that is optimized using Grid Search

Created on Nov 28, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.4 -m mind.thought_classifier [file_name] [document_column_name] [label_column_name]
# python3.4 -m mind.thought_classifier data/input/Thoughts.csv Thought Type

#####################################################

import csv
import sys
import logging
import os
from random import random

from pprint import pprint
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.base import TransformerMixin

def split_data(file_name):
	"""Divides the training set into parts for testing and training."""
	
	if not os.path.isfile(file_name):
		logging.error("Please provide a set of labeled thoughts to"\
			+ "build the classifier on")

	# Load Data
	thoughts, labels = load_data(file_name)

	# Split into training and testing sets
	if len(thoughts) < 100:
		logging.error("Not enough labeled data to create a model from")

	return thoughts, labels

def load_data(file_name):
	"""Loads human labeled data from a file."""

	thoughts, labels = [], []
	human_labeled_file = open(file_name, encoding='utf-8', errors="replace")
	human_labeled = list(csv.DictReader(human_labeled_file, delimiter=','))
	human_labeled_file.close()
	doc_col_name = sys.argv[2]
	label_col_name = sys.argv[3]

	for i in range(len(human_labeled)):
		thoughts.append(human_labeled[i][doc_col_name])
		labels.append(human_labeled[i][label_col_name].upper())

	return thoughts, labels

def build_model(thoughts, labels):
	"""Creates a classifier using the training set and then scores the
	result."""

	pipeline = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier())
	])

	parameters = {
		'vect__max_df': (0.05, 0.10, 0.25, 0.5),
		'vect__max_features': (1000, 2000, 3000, 4000, 5000, 6000),
		'vect__ngram_range': ((1, 1), (1,2)),  # unigrams or bigrams
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.00001, 0.0000055, 0.000001),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (10, 50, 80)
	}

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3, cv=3)
	grid_search.fit(thoughts, labels)
	score = grid_search.score(thoughts, labels)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	print("Actual Score: " + str(score))

	# Test Model
	#test_model("data/input/ThoughtsNov28.txt", grid_search)

	# Save Model
	joblib.dump(grid_search, 'models/thought_classifier_' + str(score) + '.pkl', compress=3)

def test_model(file_to_test, model):
	"""Tests our classifier."""
	thoughts, labels = load_data([], [], file_to_test)
	score = model.score(thoughts, labels)

	print(file_to_test, " Score: ", score)

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""
	
	if len(command_line_arguments) == 4:
		data = split_data(command_line_arguments[1])
		build_model(*data)
	else:
		print("Incorrect number of arguments")

if __name__ == "__main__":
	run_from_command_line(sys.argv)
