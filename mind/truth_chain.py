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