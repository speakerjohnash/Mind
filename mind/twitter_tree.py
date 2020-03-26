#!/usr/local/bin/python3

"""This module integrates the Twitter API and generates 
a json lookup for the knowledge tree visualization

Created on Jan 16, 2020
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3 -m mind.twitter_tree

#####################################################

import os
import sys
import json
import csv
from collections import Counter as count

import pandas as pd
import tweepy
import gensim
from gensim.models import word2vec

from mind.tools import get_write_func, load_dataframe

search_terms = ['#gameb', 'sensemaking', 'metamodernism', '"memetic mediator"', '"meaning crisis"', 'non-rivalrous', '#transitionb']

def twitter_connect():
	"""Connect to Twitter API"""

	# Load credentials from json file
	with open("config/twitter_credentials.json", "r") as file:
		creds = json.load(file)

	auth = tweepy.OAuthHandler(creds["CONSUMER_KEY"], creds["CONSUMER_SECRET"])
	auth.set_access_token(creds["ACCESS_TOKEN"], creds["ACCESS_SECRET"])

	try:
		api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
		print("Authentication OK")
	except:
		print("Error during authentication")

	return api

def twitter_tree(api):
	"""Build JSON for Tree of Knowledge"""

	column_names = ['username', 'tweet', 'tweet_id', 'created']
	out_file = "data/twitter_sensemaking_L.csv"
	user_ids = []
	output = []

	write_output = get_write_func(out_file, column_names)

	# Get User IDs of accounts mentioning phrase or hashtag
	output, user_id = search_for_terms(output, user_ids)

	# Save focus tweets
	write_output(output)

	# Select most vocal users
	mention_count = count(user_ids).most_common() # Count Users
	user_ids = list(set(user_ids)) # Get Unique Users

	# Get recent statuses from user IDs
	for user_id in user_ids:
		print("Getting contextual tweets from: " + user_id)
		for tweet in tweepy.Cursor(api.user_timeline, id=user_id).items(25):
			output.append([tweet.user.screen_name, tweet.text.encode("utf-8"), tweet.id_str, tweet.created_at])

	# Get tweets containing key terms from known users
	output = get_focus_tweets(user_ids, output)

	# Save context tweets
	write_output(output) 

	# Remove duplicates
	consolidate_tweets(out_file)

	# Train word2vec on statuses?
	similarity_lookup = {}

	return similarity_lookup

def search_for_terms(output, user_ids):
	"""Broad search for terms"""

	for term in search_terms:

		print("Searching for: " + term)
		tweetCount, max_id = 0, -1

		while tweetCount < 1000:

			try:

				if (max_id <= 0):
					new_tweets = api.search(q=term, lang="en", count=100)
				else:
					new_tweets = api.search(q=term, lang="en", count=100, max_id=str(max_id - 1))

				if new_tweets:
					for tweet in new_tweets:
						user_ids.append(tweet.user.screen_name)
						output.append([tweet.user.screen_name, tweet.text.encode("utf-8"), tweet.id_str, tweet.created_at])

				tweetCount += len(new_tweets)

				if len(new_tweets) > 0:
					max_id = new_tweets[-1].id
				else:
					break

			except tweepy.TweepError as e:
				print("Error : " + str(e))
				break

		print("Found " + str(tweetCount) + " tweets about " + term)

	return output, search_terms

def get_focus_tweets(user_ids, output=[]):
	"""Get tweets containing key terms from known users"""

	for user_id in user_ids:
		print("Getting focus tweets from: " + user_id)
		try:
			for tweet in tweepy.Cursor(api.user_timeline, id=user_id).items(500):
				for term in search_terms:
					if term in tweet.text:
						print(tweet.text + "\n")
						output.append([tweet.user.screen_name, tweet.text.encode("utf-8"), tweet.id_str, tweet.created_at])
						break
		except tweepy.TweepError as e:
			print("Error : " + str(e))

	return output

def consolidate_tweets(filename):
	"""Remove duplicate tweets"""

	search_terms = ['#gameb', 'sensemaking', 'metamodernism', 'memetic mediator', 'meaning crisis', 'non-rivalrous', '#transitionb']
	counter = dict(zip(search_terms, [0] * len(search_terms)))
	file_exists = os.path.isfile(filename)

	if file_exists:
		df = load_dataframe(filename, sep=",", quoting=csv.QUOTE_ALL)

	# Remove duplicates
	print("Rows before consolidation: " + str(df.shape[0]))

	df.drop_duplicates(subset="tweet_id", keep="first", inplace=True)

	print("Rows after consolidation: " + str(df.shape[0]))

	# Subsample tweets from users

	# Get counts of tweets per user minus tweets containing key words

	counts = df.username.value_counts()
	counts_dict = counts.to_dict()
	total_count = 0

	for user in counts_dict.keys():

		tweeter = df.loc[df['username'] == user]

		for tweet in tweeter['tweet'].to_list():
			for term in search_terms:
				if term in tweet:
					counter[term] += 1
					total_count += 1
					break

	print("Tweets containing search terms: " + str(total_count))
	print("Term count: ")
	print(counter)

	# Write to file
	df.to_csv(os.path.splitext(filename)[0] + "_consolidated.csv", index=False)

if __name__ == "__main__":
	api = twitter_connect()
	similarity_lookup = twitter_tree(api)
