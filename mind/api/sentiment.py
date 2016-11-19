#!/usr/local/bin/python3.4

"""This module defines the Prophet Mind Sentiment Analysis web service API

Created on Jul 07, 2015
@author: Matthew Sevrens
"""

import concurrent.futures
import json

from tornado import gen
from tornado_json.requesthandlers import APIHandler

from scipy.interpolate import interp1d

from mind.api import schema

HL_PATTERN = re.compile(r"HL \d+")
HAPPY_PATTERN = re.compile(r"#happy:? \d+\.?\d?")
MOOD_PATTERN = re.compile(r"#mood:? \d+\.?\d?")

def HL(scale, x):
	match = re.search(HL_PATTERN, x["Thought"])
	if match:
		return scale(float(match.group().split(" ")[1]))
	else: 
		return ""

def parse_mood(scale, x):
	happy_match = re.search(HAPPY_PATTERN, x["Thought"].lower())
	mood_match = re.search(MOOD_PATTERN, x["Thought"].lower())
	if happy_match:
		value = float(happy_match.group().split(" ")[1])
		value = value if value <= 10 else 10
		return scale(value)
	elif mood_match:
		value = float(mood_match.group().split(" ")[1])
		value = value if value <= 10 else 10
		return scale(value)
	else:
		return ""

def process_by_day(data):
	"""Process data by data and return in proper format"""

	output = {"days" : []}
	scale = interp1d([0, 10], [-1, 1])

	for day in data["days"]:

		thoughts = day["thoughts"]
		moods = []

		for thought in thougths:
			mood = parse_mood(scale, thought)
			if mood != "":
				moods.append(mood)

		if len(moods) > 0:
			average_mood = sum(moods) / float(len(moods))
			today = {"mood": average_mood, "date": day["date"]}
			output["days"].append(today)
		else:
			continue

	return output

class Sentiment_Analysis(APIHandler):
	"""This class handles Sentiment Analysis for Prophet"""

	with open("schemas/sentiment/schema_input.json") as data_file:
		schema_input = json.load(data_file)

	with open("schemas/sentiment/example_input.json") as data_file:
		example_input = json.load(data_file)

	with open("schemas/sentiment/schema_output.json") as data_file:
		schema_output = json.load(data_file)

	with open("schemas/sentiment/example_output.json") as data_file:
		example_output = json.load(data_file)

	@schema.validate(
		input_schema=schema_input,
		input_example=example_input,
		output_schema=schema_output,
		output_example=example_output
	)

	def post(self):
		"""Handle post requests"""
		data = json.loads(self.request.body.decode())
		results = process_by_day(data)
		return results

	def get(self):
		"""Handle get requests"""
		return None

if __name__ == "__main__":
	print("This module is a Class; it should not be run from the console.")