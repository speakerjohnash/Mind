#!/usr/local/bin/python3.4

"""This module defines the Meerkat web service API

Created on Jun 20, 2015
@author: Matthew Sevrens
"""

import concurrent.futures
import json

from tornado import gen
from tornado_json.requesthandlers import APIHandler

class Mind_API(APIHandler):
	"""This class is the Prophet Mind API"""

	thread_pool = concurrent.futures.ThreadPoolExecutor(8)

	with open("mind/schema_input.json") as data_file:
		schema_input = json.load(data_file)

	with open("mind/example_input.json") as data_file:
		example_input = json.load(data_file)

	with open("mind/schema_output.json") as data_file:
		schema_output = json.load(data_file)

	with open("mind/example_output.json") as data_file:
		example_output = json.load(data_file)

	@schema.validate(
		input_schema=schema_input,
		input_example=example_input,
		output_schema=schema_output,
		output_example=example_output
	)

	@gen.coroutine
	def post(self):
		"""Handle post requests asynchonously"""
		data = json.loads(self.request.body.decode())
		classifier = {}
		results = yield self.thread_pool.submit(classifier, data)
		return results

	def get(self):
		"""Handle get requests"""
		return None

if __name__ == "__main__":
	print("This module is a Class; it should not be run from the console.")