#!/usr/local/bin/python3

"""This module demonstrates the scoring function
behind Prophet

Created on Jan 09, 2017
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.truth_chain
# http://www.metaculus.com/help/scoring

#####################################################

import logging
import math
import sys

import numpy as np

from mind.tools import load_dict_list

logging.basicConfig(level=logging.INFO)

def run_from_command_line():
	"""Run module from command line"""

	scores = load_dict_list("data/input/ubs_votingapi_vote.csv")
	truth_scores = [x for x in scores if x["tag"] == "Prescience"]

	# TODO
	# Load associated thoughts and merge data

	print(len(truth_scores))

	#for score in truth_scores:
	#	print(score)

	# TODO
	# Temporal CNN for encoding thoughts using diluted convolutions
	# Append output of cnn to encodings of [time of day]:
	# 
	# df['sin_time'] = np.sin(2*np.pi*df.seconds/seconds_in_day)
	# df['cos_time'] = np.cos(2*np.pi*df.seconds/seconds_in_day)
	#
	# [day_of_year]
	#
	# [truth (0, 1)]
	#
	# [dissonance]
	#
	# [temporal_focus {-1, 0, 1}] 
	#
	# [sentiment (-1, 1)]
	#
	# [speaker]
	# Speaker embedding 

	# TODO
	# Take concatenated feautures and feed thoughts sequentially into an LSTM
	# Use the output state at each step as an input to a decoder
	# Attempt to reconctruct the input thoughts, and to generate subsequent thoughts
	# yet to be fed into the network
	# 
	# The intial loss is just the reconstruction of present, past and future thoughts
	# Some thoughts may have words modified via a thesaurus for the reconstruction signal
	#
	# After output sounds somewhat logical the loss will switch to dissonance fedback from the crowd
	# on thoughts generated
	# 
	# Perhaps a dual loss of truth and dissonance can be used. Maximize truth and minimize dissonance simultaneously

	# TODO
	# Analytics
	# How many thoughts have more than one truth vote?
	# How many unique users have logged votes?
	# How many thoughts have dissonance between users?
	# What is the distribution of truth votes?

	# Should reward more: correct contrarianism
	# Should reward less: correct alignment with the crowd
	# Should penalize more: incorrect contrarianism
	# Should penalize less: incorrect alignment with the crowd

	# One vote per day

if __name__ == "__main__":
	run_from_command_line()