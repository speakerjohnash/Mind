#!/usr/local/bin/python3.3

"""This module generates and trains classificaton models for
using SciKit Learn. Specifically it uses a Stochastic 
Gradient Descent classifier that is optimized using Grid Search

Created on Nov 28, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 thoughtClassifier.py [file_name] [label_column_name] [document_column_name]

#####################################################

import csv
import sys
import logging
import os
from random import random

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.base import TransformerMixin

def split_data(labeled_thoughts):
	"""Divides the training set into parts for testing and training."""
	if not os.path.isfile(labeled_thoughts):
		logging.error("Please provide a set of labeled thoughts to"\
			+ "build the classifier on")

	thoughts = []
	labels = []

	# Load Data
	thoughts, labels = load_data(thoughts, labels, labeled_thoughts)

	# Append More
	#thoughts, labels = load_data(thoughts, labels, "data/input/ThoughtsNov28.csv")

	print("NUMBER OF THOUGHTS: ", len(thoughts))

	# Split into training and testing sets
	if len(thoughts) < 100:
		logging.error("Not enough labeled data to create a model from")

	trans_train, trans_test, labels_train,\
		labels_test = train_test_split(thoughts, labels, test_size=0.2)

	return trans_train, trans_test, labels_train, labels_test

def load_data(thoughts, labels, file_name):
	"""Loads human labeled data from a file."""

	human_labeled_file = open(file_name, encoding='utf-8', errors="replace")
	human_labeled = list(csv.DictReader(human_labeled_file, delimiter=','))
	human_labeled_file.close()
	label_col_name = sys.argv[2]
	doc_col_name = sys.argv[3]

	for i in range(len(human_labeled)):
		thoughts.append(human_labeled[i][doc_col_name])
		labels.append(human_labeled[i][label_col_name].upper())

	return thoughts, labels

def build_model(trans_train, trans_test, labels_train, labels_test):
	"""Creates a classifier using the training set and then scores the
	result."""

	pipeline = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier())
	])

	parameters = {
		'vect__max_df': (0.25, 0.35),
		'vect__max_features': (None, 9000),
		'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': (0.0000055, 0.000008),
		'clf__penalty': ('l2', 'elasticnet'),
		'clf__n_iter': (50, 80)
	}

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=3, cv=3)
	grid_search.fit(trans_train, labels_train)
	score = grid_search.score(trans_test, labels_test)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	print("Actual Score: " + str(score))

	# Test Model
	#test_model("data/input/ThoughtsNov28.txt", grid_search)

	# Save Model
	joblib.dump(grid_search, 'model_' + str(score) + '.pkl', compress=3)

def test_model(file_to_test, model):
	"""Tests our classifier."""
	thoughts, labels = load_data([], [], file_to_test)
	score = model.score(thoughts, labels)

	print(file_to_test, " Score: ", score)

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""
	if len(command_line_arguments) == 4\
	and os.path.isfile(command_line_arguments[1]):
		trans_train, trans_test, labels_train, labels_test =\
			split_data(labeled_thoughts=command_line_arguments[1])
	else:
		trans_train, trans_test, labels_train, labels_test = split_data()
	build_model(trans_train, trans_test, labels_train, labels_test)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
