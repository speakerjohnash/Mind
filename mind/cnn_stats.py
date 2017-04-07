#!/usr/local/bin/python3.3

"""This module reports stats on a trained model

Created on Jan 15, 2016
@author: Matthew Sevrens
"""

####################### USAGE ##########################

# python3 -m mind.cnn_stats
# -model <path_to_classifier> 
# -data <path_to_testdata> 
# -map <path_to_label_map> 
# -label <ground_truth_label_key> 

########################################################

import argparse
import csv
import logging
import numpy as np
import os
import pandas as pd
import sys

from mind.load_model import get_cnn_by_path
from mind.tools import load_json, get_write_func

def parse_arguments(args):
	""" Create the parser """

	parser = argparse.ArgumentParser(description="Test a cnn and return performance statistics")
	parser.add_argument('--model', '-model', required=True, help='Path to the model under test')
	parser.add_argument('--testdata', '-data', required=True, help='Path to the test data')
	parser.add_argument('--label_map', '-map', required=True, help='Path to a label map')
	parser.add_argument('--label_key', '-label', required=True, help="Header name of the ground truth label column")

	parser.add_argument("-d", "--debug", help="Show 'debug'+ level logs", action="store_true")
	parser.add_argument("-v", "--info", help="Show 'info'+ level logs", action="store_true")
	
	return parser.parse_args(args)

def compare_label(*args, **kwargs):
	"""Calculate accuracy"""

	machine, cnn_column, human_column, conf_mat, num_labels = args[:]
	doc_key = "Thought"
	unpredicted = []
	needs_hand_labeling = []
	correct = []
	mislabeled = []

	# Test Each Machine Labeled Row
	for machine_row in machine:

		# Update Confusion Matrix
		if machine_row['ACTUAL_INDEX'] is None:
			pass
		else:
			column = machine_row['PREDICTED_INDEX']
			row = machine_row['ACTUAL_INDEX']
			conf_mat[row][column] += 1

		# Record Results
		if machine_row[cnn_column] == "":
			unpredicted.append([machine_row[doc_key], machine_row[human_column]])
			continue

		# Identify unlabeled points
		if not machine_row[human_column]:
			needs_hand_labeling.append(machine_row[doc_key])
			continue

		# Predicted label matches human label
		if machine_row[cnn_column] == machine_row[human_column]:
			correct.append([machine_row[doc_key], machine_row[human_column]])
			continue

		mislabeled.append([machine_row[doc_key], machine_row[human_column], machine_row[cnn_column]])

	return mislabeled, correct, unpredicted, needs_hand_labeling, conf_mat

def count_thoughts(csv_file):
	"""count number of thoughts in csv_file"""
	with open(csv_file) as temp:
		reader = csv.reader(temp, delimiter='|')
		_ = reader.__next__()
		return sum([1 for i in reader])

def get_classification_report(confusion_matrix_file, label_map):
	"""Produce a classification report for a particular confusion matrix"""

	df = pd.read_csv(confusion_matrix_file)
	rows, cols = df.shape

	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")

	# First order calculations
	true_positive = pd.DataFrame(df.iat[i, i] for i in range(rows))
	col_sum = pd.DataFrame(df.sum(axis=1))
	false_positive = pd.DataFrame(pd.DataFrame(df.sum(axis=0)).values - true_positive.values, columns=true_positive.columns)
	false_negative = pd.DataFrame(pd.DataFrame(df.sum(axis=1)).values - true_positive.values, columns=true_positive.columns)
	true_negative = pd.DataFrame([df.drop(str(i), axis=1).drop(i, axis=0).sum().sum() for i in range(rows)])

	# Second order calculations
	accuracy = true_positive.sum() / df.sum().sum()
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)
	specificity = true_negative / (true_negative + false_positive)

	# Third order calculation
	f_measure = 2 * precision * recall / (precision + recall)

	# Write out the classification report
	label = pd.DataFrame(label_map, index=[0]).transpose()
	label.index = label.index.astype(int)
	label = label.sort_index()
	num_labels = len(label_map)
	label.index = range(num_labels)

	# Define Report Values
	features = {
		"Accuracy": accuracy,
		"Class": label,
		"True Positive": true_positive,
		"True Negative": true_negative,
		"False Positive": false_positive,
		"False Negative": false_negative,
		"Precision": precision,
		"Recall": recall,
		"Specificity": specificity,
		"F Measure": f_measure
	}

	# Craft the Report
	classification_report = pd.concat(features.values(), axis=1)
	classification_report.columns = features.keys()

	logging.debug("Classification Report:\n{0}".format(classification_report))
	logging.info("Accuracy is: {0}".format(classification_report.iloc[0]["Accuracy"]))
	report_path = 'data/CNN_stats/classification_report.csv'
	logging.info("Classification Report saved to: {0}".format(report_path))
	classification_report.to_csv(report_path, index=False)

# Main
def main_process(args):
	"""This is the main stream"""

	machine_label_key = 'PREDICTED_CLASS'
	doc_key = "Thought"
	human_label_key = args.label_key
	reader = pd.read_csv(args.testdata, na_filter=False, chunksize=1000)
	total_thoughts = count_thoughts(args.testdata)
	processed, chunk_count = 0.0, 0
	label_map = load_json(args.label_map)
	reversed_label_map = {}

	get_key = lambda x: x['label'] if isinstance(x, dict) else x
	label_map = dict(zip(label_map.keys(), map(get_key, label_map.values())))
	num_labels = len(label_map)
	class_names = list(label_map.values())
	reversed_label_map = dict(zip(label_map.values(), label_map.keys()))

	confusion_matrix = [[0 for i in range(num_labels + 1)] for j in range(num_labels)]
	classifier = get_tf_cnn_by_path(args.model, args.label_map)

	# Prepare for data saving
	path = "data/CNN_stats/"
	os.makedirs(path, exist_ok=True)
	write_mislabeled = get_write_func(path + "mislabeled.csv", ['THOUGHT', 'ACTUAL', 'PREDICTED'])
	write_correct = get_write_func(path + "correct.csv", ['THOUGHT', 'ACTUAL'])
	write_unpredicted = get_write_func(path + "unpredicted.csv", ["THOUGHT", 'ACTUAL'])
	write_needs_hand_labeling = get_write_func(path + "need_labeling.csv", ["THOUGHT"])

	logging.info("Total number of thoughts: {0}".format(total_thoughts))
	logging.info("Testing begins.")

	for chunk in reader:
		processed += len(chunk)
		my_progress = str(round(((processed/total_thoughts) * 100), 2)) + '%'
		logging.info("Evaluating {0} of the testset".format(my_progress))
		logging.warning("Testing chunk {0}.".format(chunk_count))
		thoughts = chunk.to_dict('records')
		machine_labeled = classifier(thoughts, doc_key=doc_key, label_key=machine_label_key)

		# Add Indexes for Labels
		for item in machine_labeled:

			if item[human_label_key] == "":
				item['ACTUAL_INDEX'] = None
				continue

			item['ACTUAL_INDEX'] = int(reversed_label_map[item[human_label_key]])
			item['PREDICTED_INDEX'] = int(reversed_label_map[item[machine_label_key]])

		results = compare_label(machine_labeled, machine_label_key, human_label_key, confusion_matrix, num_labels, doc_key=doc_key)
		mislabeled, correct, unpredicted, needs_hand_labeling, confusion_matrix = results

		# Save
		write_mislabeled(mislabeled)
		write_correct(correct)
		write_unpredicted(unpredicted)
		write_needs_hand_labeling(needs_hand_labeling)

		chunk_count += 1

	# Make a Square Confusion Matrix Dataframe
	df = pd.DataFrame(confusion_matrix)
	df = df.drop(df.columns[[-1]], axis=1)
	rows, cols = df.shape

	# Check if Confusion Matrix is a Square
	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")

	# Save the confusion matrix out to a file
	confusion_matrix_path = 'data/CNN_stats/confusion_matrix.csv'
	logging.debug("Rows: {0}, Columns {1}".format(rows, cols))
	df.to_csv('data/CNN_stats/confusion_matrix.csv', index=False)
	logging.info("Confusion matrix saved to: {0}".format(confusion_matrix_path))
	get_classification_report(confusion_matrix_path, label_map)

if __name__ == "__main__":

	args = parse_arguments(sys.argv[1:])
	log_format = "%(asctime)s %(levelname)s: %(message)s"

	if args.debug:
		logging.basicConfig(format=log_format, level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(format=log_format, level=logging.INFO)
	else:
		logging.basicConfig(format=log_format, level=logging.WARNING)

	main_process(args)
