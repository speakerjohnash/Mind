#!/usr/local/bin/python3.3

"""This module processes a sample of thoughts and outputs 
a model which will selectively tokenize bigrams and unigrams

Created on Jan 21, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m mind.bigram_tokenizer [thought_sample] 
# python3.3 -m mind.bigram_tokenizer data/input/Thoughts.csv

# Required Columns: 
# Thought

#####################################################

import contextlib
import csv
import pandas as pd
import sys

from gensim.models import Phrases

from various_tools import safe_print, safe_input

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

	sample_included = sample.endswith('.txt')

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

	reader = pd.read_csv(cla[1], chunksize=5000, na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep=',', error_bad_lines=False)

	# Process Thoughts
	for chunk in reader:

		thoughts = [row["Thought"].split(" ") for i, row in chunk.iterrows()]

		if first_chunk:
			bigrams = Phrases(thoughts, max_vocab_size=5000)
		else:
			bigrams.add_vocab(thoughts)

	bigrams.save("models/bigram_model")

	#for index, value in df['Thought'].iteritems():
	#	safe_print(value)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
