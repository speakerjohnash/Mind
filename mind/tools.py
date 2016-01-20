#!/usr/local/bin/python3.3

"""A collection of functions to be called by multiple modules throughout 
Prophet Mind

Created on Jan 21, 2015
@author: Matthew Sevrens
"""

import pandas as pd
import csv
import json
import os
import sys

import numpy as np

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter, quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def to_stdout(string, errors="replace"):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open('data/output/' + filename, 'w') as fp:
		json.dump(obj, fp, indent=4)

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""
	print(*(to_stdout(str(o), errors) for o in objs))

def safe_input(prompt=""):
	"""Safely input a string"""

	try:
		result = input(prompt)
		return result
	except KeyboardInterrupt:
		sys.exit()
	except:
		return ""

def progress(i, my_list, message=""):
	"""Display progress percent in a loop"""
	my_progress = (i / len(my_list)) * 100
	my_progress = str(round(my_progress, 1)) + "% " + message
	sys.stdout.write('\r')
	sys.stdout.write(my_progress)
	sys.stdout.flush()

def load_params(filename):
	"""Load a set of parameters provided a filename"""

	input_file = open(filename, encoding='utf-8')
	params = json.loads(input_file.read())
	input_file.close()

	return params

def safely_remove_file(filename):
	"""Safely removes a file"""
	print("Removing {0}".format(filename))
	try:
		os.remove(filename)
	except OSError:
		print("Unable to remove {0}".format(filename))
	print("File removed.")
	
def get_write_func(filename, header):
    
    file_exists = False
    
    def write_func(data):
        nonlocal file_exists
        mode = "a" if file_exists else "w"
        add_head = False if file_exists else header
        df = pd.DataFrame(data)
        df.to_csv(filename, mode=mode, index=False, header=add_head)
        file_exists = True
    
    return write_func

if __name__ == "__main__":
	print("This module is a library that contains useful functions; it should not be run from the console.")
