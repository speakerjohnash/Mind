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
import csv

import pandas as pd
import tweepy
import gensim
from gensim.models import word2vec

from mind.tools import get_write_func

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

	user_names = []
	outfile = []

	output_file = get_write_func("data/twitter_sensemaking.csv", ['username', 'tweet', 'tweet_id', 'created'])

	# Get User IDs of accounts mentioning phrase or hashtag
	for tweet in api.search(q="#gameB", lang="en", count=5):
		user_names.append(tweet.user.name)
		# print(f"{tweet.user.name} : {tweet.text} : {tweet.id_str} : {tweet.created_at}")

	# Select users with most references to provided concept

	# Get statuses from user IDs
	for user_name in user_names:
		for tweet in tweepy.Cursor(api.user_timeline, id=user_name).items(5):
			outfile.append([tweet.user.name, tweet.text.encode("utf-8"), tweet.id_str, tweet.created_at])

	# Save tweets to csv?

	# Train word2vec on statuses?

	similarity_lookup = {}

	return similarity_lookup

if __name__ == "__main__":
	api = twitter_connect()
	similarity_lookup = twitter_tree(api)
