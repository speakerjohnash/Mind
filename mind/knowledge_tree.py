import os
import sys
import re
import nltk
import math

from sklearn.feature_extraction.text import CountVectorizer

class MySentences(object):
 	    def __init__(self, dirname):
 	        self.dirname = dirname
 	
 	    def __iter__(self):
 	        for fname in os.listdir(self.dirname):
 	            for line in open(os.path.join(self.dirname, fname)):
 	                yield line.split()

def to_stdout(string, errors='replace'):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""

	print(*(to_stdout(str(o), errors) for o in objs))

def progress(i, list, message=""):
	"""Display progress percent in a loop"""

	progress = (i / len(list)) * 100
	progress = str(round(progress, 1)) + "% " + message
	sys.stdout.write('\r')
	sys.stdout.write(progress)
	sys.stdout.flush()                

thoughts = MySentences('/Users/Matt/Desktop/Prophet/thoughts')

# Process Thoughts
