#!/usr/local/bin/python3.3

"""This module generates semantic vector representations
from a given corpus of documents

Created on Nov 26, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.4 -m mind.semanticThoughts [file_name]

#####################################################

import csv
import sys

from gensim.models import Word2Vec
from gensim.utils import tokenize

class documentGenerator(object):
	"""A memory efficient document loader"""

	def __init__(self, file_name, doc_column):
		df = pd.read_csv(file_name, na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep=',', error_bad_lines=False)
		self.docs = df[doc_column]

	def __iter__(self):
		for doc in self.docs.iteritems():
			yield tokenize(doc[1].lower())

def run_from_command_line(command_line_arguments):
	"""Runs the module when invoked from the command line."""

	#documents = documentGenerator(sys.argv[1], "Thought")

	input_file = open(sys.argv[1], encoding='utf-8', errors='replace')
	thoughts = list(csv.DictReader(input_file, delimiter=','))
	input_file.close()

	thoughts = [t['Thought'].split() for t in thoughts]
	model = Word2Vec(thoughts)
	model.save('./models/word2vec_thoughts')

if __name__ == "__main__":
	run_from_command_line(sys.argv)
