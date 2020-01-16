#!/usr/local/bin/python3

"""This module integrates the Twitter API and generates 
a json lookup for the knowledge tree visualization

Created on Jan 16, 2020
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.twitter_tree

#####################################################

import sys
import json

import gensim
from gensim.models import word2vec

def twitter_connect():
	"""Connect to Twitter API"""

	api = None

	return api

def twitter_tree(api):
	"""Build JSON for Tree of Knowledge"""

	similarity_lookup = {}

	return similarity_lookup

if __name__ == "__main__":
	api = twitter_connect
	similarity_lookup = twitter_tree(api)
