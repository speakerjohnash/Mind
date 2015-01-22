#!/usr/local/bin/python3.4

"""This module processes a sample of thoughts and outputs 
a model which will selectively tokenize bigrams and unigrams

Created on Jan 21, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.4 -m mind.bigram_tokenizer [thought_sample] 
# python3.4 -m mind.bigram_tokenizer data/input/Thoughts.csv

# Required Columns: 
# Thought

#####################################################

import contextlib
import csv
import pandas as pd
import sys

from gensim.models import Phrases

from mind.tools import safe_print, safe_input

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

def verify_arguments():
	"""Verify Usage"""

	sufficient_arguments = (len(sys.argv) == 2)

	if not sufficient_arguments:
		safe_print("Insufficient arguments. Please see usage")
		sys.exit()

	sample = sys.argv[1]

	sample_included = sample.endswith('.txt') or sample.endswith('.csv')

	if not sample_included:
		safe_print("Erroneous arguments. Please see usage")
		sys.exit()

def add_local_params(params):
	"""Adds additional local params"""

	params["generate_common_bigrams"] = {
	}

	return params

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = {}
	params = add_local_params(params)
	first_chunk = True
	corpus = []

	reader = pd.read_csv(cla[1], chunksize=5000, na_filter=False, encoding="utf-8", sep=',', error_bad_lines=False)

	# Process Thoughts
	for chunk in reader:

		thoughts = []

		for i, row in chunk.iterrows():
			thoughts.append(row["Thought"].lower().split(" "))
			corpus.append(row["Thought"].lower().split(" "))

		if first_chunk:
			bigrams = Phrases(thoughts, max_vocab_size=10000)
		else:
			bigrams.add_vocab(thoughts)

	bigrams.save("models/bigram_model")

	for thought in corpus:
		safe_print(bigrams[thought])

if __name__ == "__main__":
	run_from_command_line(sys.argv)
