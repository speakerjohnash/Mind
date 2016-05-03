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

from mind.load_model import get_tf_cnn_by_path
from mind.tools import load_json, get_write_func

def parse_arguments(args):
	""" Create the parser """

	parser = argparse.ArgumentParser(description="Test a cnn and return performance statistics")
	parser.add_argument('--model', '-model', required=True, help='Path to the model under test')
	parser.add_argument('--testdata', '-data', required=True, help='Path to the test data')
	parser.add_argument('--label_map', '-map', required=True, help='Path to a label map')
	parser.add_argument('--label_key', '-label', required=True, type=lambda x: x.upper(), help="Header name of the ground truth label column")

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
		# Update conf_mat
		# predicted_label is None if a predicted subtype is ""
		if machine_row['ACTUAL_INDEX'] is None:
			pass
		elif machine_row['PREDICTED_INDEX'] is None:
			column = num_labels
			row = machine_row['ACTUAL_INDEX'] - 1
			conf_mat[row][column] += 1
		else:
			column = machine_row['PREDICTED_INDEX'] - 1
			row = machine_row['ACTUAL_INDEX'] - 1
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

def count_transactions(csv_file):
	"""count number of transactions in csv_file"""
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

	#Convert to 0-indexed confusion matrix
	df.rename(columns=lambda x: int(x) - 1, inplace=True)
	#First order calculations
	true_positive = pd.DataFrame(df.iat[i, i] for i in range(rows))
	col_sum = pd.DataFrame(df.sum(axis=1))
	false_positive = pd.DataFrame(pd.DataFrame(df.sum(axis=0)).values - true_positive.values,
		columns=true_positive.columns)
	false_negative = pd.DataFrame(pd.DataFrame(df.sum(axis=1)).values - true_positive.values,
		columns=true_positive.columns)
	true_negative = pd.DataFrame(
		[df.drop(i, axis=1).drop(i, axis=0).sum().sum() for i in range(rows)])

	#Second order calculations
	accuracy = true_positive.sum() / df.sum().sum()
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)
	specificity = true_negative / (true_negative + false_positive)

	#Third order calculation
	f_measure = 2 * precision * recall / (precision + recall)

	#Write out the classification report
	label = pd.DataFrame(label_map, index=[0]).transpose()
	label.index = label.index.astype(int)
	label = label.sort_index()
	num_labels = len(label_map)
	label.index = range(num_labels)

	#Create a classification report
	feature_list = [accuracy, label, true_positive, false_positive,
		false_negative, true_negative, precision, recall, specificity,
		f_measure]
	feature_labels = ["Accuracy", "Class", "True Positive", "False Positive",
		"False Negative", "True Negative", "Precision", "Recall", "Specificity",
		"F Measure"]
	#Craft the report
	classification_report = pd.concat(feature_list, axis = 1)
	classification_report.columns = feature_labels
	#Setting rows to be 1-indexed
	classification_report.index = range(1, rows + 1)

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
	reader = load_piped_dataframe(args.testdata, chunksize=1000)
	total_transactions = count_transactions(args.testdata)
	processed = 0.0
	label_map = load_json(args.label_map)

	get_key = lambda x: x['label'] if isinstance(x, dict) else x
	label_map = dict(zip(label_map.keys(), map(get_key, label_map.values())))
	num_labels = len(label_map)
	class_names = list(label_map.values())

	# Create reversed label map and check it there are duplicate keys
	reversed_label_map = {}
	for key, value in label_map.items():
		if class_names.count(value) > 1:
			reversed_label_map[value] = sorted(reversed_label_map.get(value, []) + [int(key)])
		else:
			reversed_label_map[value] = int(key)

	confusion_matrix = [[0 for i in range(num_labels + 1)] for j in range(num_labels)]
	classifier = get_tf_cnn_by_path(args.model, args.label_map)

	# Prepare for data saving
	path = 'data/CNN_stats/'
	os.makedirs(path, exist_ok=True)
	write_mislabeled = get_write_func(path + "mislabeled.csv", ['TRANSACTION_DESCRIPTION', 'ACTUAL', 'PREDICTED'])
	write_correct = get_write_func(path + "correct.csv", ['TRANSACTION_DESCRIPTION', 'ACTUAL'])
	write_unpredicted = get_write_func(path + "unpredicted.csv", ["TRANSACTION_DESCRIPTION", 'ACTUAL'])
	write_needs_hand_labeling = get_write_func(path + "need_labeling.csv", ["TRANSACTION_DESCRIPTION"])

	chunk_count = 0

	logging.info("Total number of transactions: {0}".format(total_transactions))
	logging.info("Testing begins.")

	for chunk in reader:
		processed += len(chunk)
		my_progress = str(round(((processed/total_transactions) * 100), 2)) + '%'
		logging.info("Evaluating {0} of the testset".format(my_progress))
		logging.warning("Testing chunk {0}.".format(chunk_count))
		transactions = chunk.to_dict('records')
		machine_labeled = classifier(transactions, doc_key=doc_key, label_key=machine_label_key)

		# Add indexes for labels
		for item in machine_labeled:
			if item[human_label_key] == "":
				item['ACTUAL_INDEX'] = None
				continue
			temp = reversed_label_map[item[human_label_key]]
			if isinstance(temp, list):
				item['ACTUAL_INDEX'] = temp[0]
			else:
				item['ACTUAL_INDEX'] = temp
			if item[machine_label_key] == "":
				item['PREDICTED_INDEX'] = None
				continue
			temp = reversed_label_map[item[machine_label_key]]
			if isinstance(temp, list):
				item['PREDICTED_INDEX'] = temp[0]
			else:
				item['PREDICTED_INDEX'] = temp

		results = compare_label(machine_labeled, machine_label_key, human_label_key, confusion_matrix, num_labels, doc_key=doc_key)
		mislabeled, correct, unpredicted, needs_hand_labeling, confusion_matrix = results

		# Save
		write_mislabeled(mislabeled)
		write_correct(correct)
		write_unpredicted(unpredicted)
		write_needs_hand_labeling(needs_hand_labeling)

		chunk_count += 1

	#Make a square confusion matrix dataframe, df
	df = pd.DataFrame(confusion_matrix)
	df = df.drop(df.columns[[-1]], axis=1)

	#Make sure the confusion matrix is a square
	rows, cols = df.shape
	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")

	#Make sure the confusion matrix is 1-indexed, to match the label_map
	df.rename(columns=lambda x: int(x) + 1, inplace=True)
	df.index = range(1, rows + 1)

	#Save the confusion matrix out to a file
	confusion_matrix_path = 'data/CNN_stats/confusion_matrix.csv'
	rows, cols = df.shape
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
