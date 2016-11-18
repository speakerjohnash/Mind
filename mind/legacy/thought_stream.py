#################### USAGE ####################################

# python3 -m mind.legacy.thought_stream [input_file] [user]
# python3 -m mind.legacy.thought_stream data/input/thought.csv matt

###############################################################


import os
import sys
import re
import math
import csv
import json
import random
import collections
import datetime

import numpy
from textblob import TextBlob, Word
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.interpolate import interp1d

from mind.tools import vectorize, word_count, safe_print, load_dict_list
from mind.load_model import get_tf_cnn_by_name

SENTIMENT = get_tf_cnn_by_name("sentiment")
HL_PATTERN = re.compile(r"HL \d+")

def HL(x):
	match = re.search(HL_PATTERN, x["Thought"])
	if match:
		return match.group().split(" ")[1]
	else: 
		return ""

def write_dict_list(dict_list, file_name, encoding="utf-8", delimiter=","):
	""" Saves a lists of dicts with uniform keys to file """

	column_order = sorted(dict_list[0])

	with open(file_name, 'w', encoding=encoding, errors='replace') as output_file:
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=column_order, extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

def compareMinds(data_a, data_b):
	"""Compares available words"""

	deltas = {}

	for key in data_a:
		a_freq = data_a[key]
		b_freq = data_b.get(key, 0)
		deltas[key] = a_freq - b_freq

	safe_print(collections.OrderedDict(sorted(deltas.items(), key=lambda t: t[1])))

def collectThoughts(thoughts):
	"""Bucket data by user for easy access"""

	thinkers = collections.defaultdict(list)

	# Split into user buckets
	for thought in thoughts:
		thinker = thought.get("Seer", "")
		thinkers[thinker].append(thought)

	return thinkers

def groupByDay(thoughts):
	"""Bucket a users thoughts by day"""

	days = collections.defaultdict(list)

	# Split by day
	for thought in thoughts:
		day = thought.get("Post date", "")[:8]
		days[day].append(thought)

	return days

def groupByWeek(thoughts):
	"""Bucket a users thoughts by week"""

	weeks = collections.defaultdict(list)

	# Split by day
	for thought in thoughts:
		day = thought.get("Post date", "")[:8]
		dt = datetime.datetime.strptime(day, '%m/%d/%y')
		weekstart = dt - datetime.timedelta(days = dt.weekday())
		weeks[weekstart.strftime('%m/%d/%y')].append(thought)

	return weeks

def processByDay(days, ken):
	"""Run the vectorizing tool on many days"""

	stream = []

	for day, thoughts in days.items():
		thoughts = [thought['Thought'] for thought in thoughts]
		word_count = vectorize(thoughts)
		daily_ken = {'Post Date' : day}
	
		# Get Daily Words
		for word, count in word_count.items():
			#word = Word(word).lemmatize()
			if word in ken:
				daily_ken[word] = count

		# Add Missing Words
		for word in ken:
			#word = Word(word).lemmatize()
			if word not in daily_ken:
				daily_ken[word] = 0

		stream.append(daily_ken)

	return stream

def buildWordStream(days, ken):
	"""Build out a word stream from daily thoughts"""

	stream = processByDay(days, sorted(ken))
	sorted_stream = sorted(stream, key=lambda k: datetime.datetime.strptime(k['Post Date'], '%m/%d/%y').date());
	write_dict_list(sorted_stream, "data/output/all_stream.csv")

def parse_mood(scale, thought):

	if "#happy" in thought["Thought"].lower():
		print(thought["Thought"])

	if "#happy" in thought["Thought"].lower():
		tokens = thought["Thought"].lower().replace("#happy", "").split(" ")
		if tokens[1].replace('.', '', 1).isdigit(): 
			return scale(float(tokens[1]))

	return None

def buildSentimentStream(days):
	"""Build out a sentiment stream from daily thoughts"""

	sentiment_stream = []
	scale = interp1d([0,11],[-1,1])

	for day, thoughts in days.items():

		votes, polarity, reported = [], [], []
		analysis = [float(x['CNN']) for x in SENTIMENT(thoughts)]

		for thought in thoughts:

			# Parse out manual tracking
			happy = parse_mood(scale, thought)

			if happy is not None:
				reported.append(happy)

			doc = TextBlob(thought["Thought"])
			polarity.append(doc.sentiment.polarity)
			vote_strings = thought["Good"].split("\n")
			good, bad, _ = vote_strings
			net_good = int(good[1:]) + int(bad)

			if net_good != 0:
				votes.append(net_good)

		if len(votes) == 0:
			votes = [0]

		average_vote = sum(votes) / float(len(votes))
		average_polarity = sum(polarity) / float(len(polarity))

		if len(reported) > 0:
			average_happy = sum(reported) / float(len(reported))
			average_mood = (average_vote + average_polarity + average_happy + average_happy + average_happy) / 5
		else:
			average_mood = average_polarity
			average_happy = 0

		daily_sentiment = {
			"average" : average_mood,
			"textBlob": average_polarity,
			"vote": average_vote,
			"mood": average_happy
		}

		daily_sentiment['Post Date'] = day
		sentiment_stream.append(daily_sentiment)

	sorted_stream = sorted(sentiment_stream, key=lambda k: datetime.datetime.strptime(k['Post Date'], '%m/%d/%y').date());
	write_dict_list(sorted_stream, "data/output/sentiment_stream.csv")

def buildTypeStream(days):
	"""Build stream data from thought type count"""

	type_stream = []

	for day, thoughts in days.items():

		day_type_counts = {
			"State" : 0,
			"Ask" : 0,
			"Predict" : 0,
			"Reflect" : 0,
			"Thought" : 0
		}

		for thought in thoughts:
			thought_type = thought['Type']
			if thought_type == "":
				thought_type = "Thought"
			day_type_counts[thought_type] += 1

		day_type_counts['Post Date'] = day
		type_stream.append(day_type_counts)

	sorted_stream = sorted(type_stream, key=lambda k: datetime.datetime.strptime(k['Post Date'], '%m/%d/%y').date());
	write_dict_list(sorted_stream, "data/output/type_stream.csv")

def buildPrivacyStream(days):
	"""Build out a privacy stream from daily thoughts"""

	privacy_stream = []

	for day, thoughts in days.items():

		privacy_counts = {
			"Private" : 0,
			"Public" : 0,
			"" : 0
		}

		for thought in thoughts:
			privacy_counts[thought['Privacy']] += 1

		privacy_counts['Post Date'] = day
		privacy_stream.append(privacy_counts)

	sorted_stream = sorted(privacy_stream, key=lambda k: datetime.datetime.strptime(k['Post Date'], '%m/%d/%y').date());
	write_dict_list(sorted_stream, "data/output/privacy_stream.csv")

def buildLookup(days, ken):
	"""Build a lookuptable for search"""

	lookup = collections.defaultdict(list)

	for day, thoughts in days.items():
		for thought in thoughts:
			word_count = vectorize([thought['Thought']])
			for word in word_count.keys():
				if word in ken:
					lookup[word].append([thought["ID"], int(datetime.datetime.strptime(day, '%m/%d/%y').timestamp())])

	with open('lookup.json', 'w') as fp:
		json.dump(lookup, fp)

def preprocess_thoughts(thoughts):
	"""Perform preprocessing steps"""

	if sys.argv[2] == "pat":
		for thought in thoughts:
			thought["mood"] = HL(thought)
			if thought["mood"] != "":
				print(thought["mood"])

	return thoughts

def run_from_command():             
	"""Run if file invoked from command line"""
	
	params = {}
	collective_thoughts = []
	thoughts = load_dict_list(sys.argv[1])
	thoughts = preprocess_thoughts(thoughts)
	thinkers = collectThoughts(thoughts)

	pat_thoughts = thinkers['patch615']
	matt_thoughts = thinkers['msevrens']
	prophet_thoughts = thinkers['prophet']
	work_thoughts = thinkers['msevrens@yodlee.com']
	leah_thoughts = thinkers['leahdaniels']

	# Automate User Selection
	if sys.argv[2] == "all":
		for thinker, thoughts in thinkers.items():
			collective_thoughts += thoughts
	if sys.argv[2] == "rossi":
		for thinker, thoughts in thinkers.items():
			collective_thoughts += thoughts
		thoughts = [thought['Thought'] for thought in collective_thoughts]
		ken = vectorize(thoughts, min_df=1)
		days = groupByWeek(collective_thoughts)
		lookup = buildLookup(days, ken)
		buildWordStream(days, ken)
		sys.exit()
	elif sys.argv[2] == "pat":
		collective_thoughts = pat_thoughts
	elif sys.argv[2] == "matt":
		collective_thoughts = matt_thoughts
	elif sys.argv[2] == "leah":
		collective_thoughts = leah_thoughts

	thoughts = [thought['Thought'] for thought in collective_thoughts]
	ken = vectorize(thoughts, min_df=1)
	days = groupByWeek(collective_thoughts)

	buildSentimentStream(days)
	buildWordStream(days, ken)
	#buildTypeStream(days)
	#buildPrivacyStream(days)

if __name__ == "__main__":
	run_from_command()
