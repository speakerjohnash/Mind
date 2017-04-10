import sys
import os
from os import listdir
from os.path import isfile, join
from itertools import groupby

import numpy as np

from nltk.corpus import comtrans
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from mind.tools import load_piped_dataframe, dict_2_json, load_json

class TranslationData():
	def __init__(self, bucket_quant, config):

		self.als = comtrans.aligned_sents('alignment-en-fr.txt')

		# Load Aligned Translation Pairs
		self.source_lines = []
		self.target_lines = []

		for al in self.als:
			self.source_lines.append(' '.join(al.mots))
			self.target_lines.append(' '.join(al.words))

		# Build word vocab

		print(("Source Sentences", len(self.source_lines)))
		print(("Target Sentences", len(self.target_lines)))

		# Build character vocab
		self.bucket_quant = bucket_quant
		self.source_vocab = self.build_char_vocab(self.source_lines)
		self.target_vocab = self.build_word_vocab()

		print(("SOURCE VOCAB SIZE", len(self.source_vocab)))
		print(("TARGET VOCAB SIZE", len(self.target_vocab)))

	def bucket_data(self):
		"""Bucket Data"""

		source_lines = []
		target_lines = []

		for i in range(len(self.source_lines)):
			source_lines.append(self.string_to_char_indices(self.source_lines[i], self.source_vocab))
			target_lines.append(self.string_to_word_indices(self.target_lines[i], self.target_vocab))

		buckets = self.create_buckets(source_lines, target_lines)

		frequent_keys = [(-len(buckets[key]), key) for key in buckets]
		frequent_keys.sort()

		print(("Source", self.char_indices_to_string( buckets[ frequent_keys[3][1] ][5][0], self.source_vocab)))
		print(("Target", self.word_indices_to_string( buckets[ frequent_keys[3][1] ][5][1], self.target_vocab)))
		
		print((len(frequent_keys)))

		return buckets, self.source_vocab, self.target_vocab, frequent_keys

	def create_buckets(self, source_lines, target_lines):
		"""Create Buckets"""
		
		bucket_quant = self.bucket_quant
		source_vocab = self.source_vocab
		target_vocab = self.target_vocab

		buckets = {}

		for i in range(len(source_lines)):
			
			source_lines[i] = np.concatenate((source_lines[i], [source_vocab['eol']]))
			target_lines[i] = np.concatenate(([target_vocab['init']], target_lines[i], [target_vocab['eol']]))
			
			sl = len(source_lines[i])
			tl = len(target_lines[i])

			new_length = max(sl, tl)

			if new_length % bucket_quant > 0:
				new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant	
			
			s_padding = np.array( [source_vocab['padding'] for ctr in range(int(sl), int(new_length)) ] )

			# Extra Padding for Training
			t_padding = np.array([target_vocab['padding'] for ctr in range(int(tl), int(new_length + 1))])
			source_lines[i] = np.concatenate([source_lines[i], s_padding])
			target_lines[i] = np.concatenate([target_lines[i], t_padding])

			if new_length in buckets:
				buckets[new_length].append((source_lines[i], target_lines[i]))
			else:
				buckets[new_length] = [(source_lines[i], target_lines[i])]

			if i%1000 == 0:
				print(("Loading", i))
			
		return buckets

	def build_char_vocab(self, sentences):
		"""Build character vocab"""

		if os.path.isfile("models/char_lookup.json"):
			return load_json("models/char_lookup.json")

		vocab = {}
		ctr = 0

		for st in sentences:
			for ch in st:
				if ch not in vocab:
					vocab[ch] = ctr
					ctr += 1

		# Add Special Characters
		vocab['eol'] = ctr
		vocab['padding'] = ctr + 1
		vocab['init'] = ctr + 2

		dict_2_json(vocab, "models/char_lookup.json")

		return vocab

	def build_word_vocab(self):
		"""Build word vocab"""

		if os.path.isfile("models/word_lookup.json"):
			return load_json("models/word_lookup.json")

		vocab = set()

		for al in self.als:
			for word in al.words:
				vocab.add(word)

		word_count = len(vocab)
		index_lookup = dict(zip(vocab, range(word_count)))

		# Add Special Characters
		index_lookup['eol'] = word_count
		index_lookup['padding'] = word_count + 1
		index_lookup['init'] = word_count + 2
		index_lookup[' '] = word_count + 3

		dict_2_json(index_lookup, "models/word_lookup.json")

		return index_lookup 

	def string_to_word_indices(self, sentence, vocab):
		"""Convert string to repeated word embeding indices
		equal the length of each word"""

		tokens = sentence.split(" ")
		indices = []

		for i, token in enumerate(tokens):
			if token in vocab:
				for ii in range(len(token)): 
					indices.append(vocab[token])
			else:
				indices.append(vocab[" "])

			if i != len(tokens):
				indices.append(vocab[" "])

		return indices

	def string_to_char_indices(self, sentence, vocab):
		"""Convert string to embedding lookup indices"""

		indices = [vocab[s] for s in sentence]

		return indices

	def word_indices_to_string(self, sentence, vocab):
		"""Collapse repeated word embedding indices to string
		and convert to string"""

		id_word = {vocab[token] : token for token in vocab}
		sentence = [x[0] for x in groupby(sentence)]
		sent = []

		for w in sentence:
			if id_word[w] == 'eol':
				break
			sent += id_word[w]

		return "".join(sent)

	def char_indices_to_string(self, sentence, vocab):
		"""Convert embedding indices to string"""

		id_ch = { vocab[ch] : ch for ch in vocab } 
		sent = []

		for c in sentence:
			if id_ch[c] == 'eol':
				break
			sent += id_ch[c]

		return "".join(sent)

	def load_batch(self, pair_list):
		"""Load batch"""

		source_sentences = []
		target_sentences = []

		for s, t in pair_list:
			source_sentences.append(s)
			target_sentences.append(t)

		return np.array(source_sentences, dtype = 'int32'), np.array(target_sentences, dtype = 'int32')

class WikiData(TranslationData):
	def __init__(self, bucket_quant, config):

		# Load Aligned Sequential Sentences
		self.source_lines = []
		self.target_lines = []

		reader = self.load_data(config)

		print(("Source Sentences", len(self.source_lines)))
		print(("Target Sentences", len(self.target_lines)))

		# Build word and character vocabs
		self.bucket_quant = config["options"]["bucket_quant"]
		self.source_vocab = self.build_char_vocab(self.source_lines)
		self.target_vocab = self.build_word_vocab()
		print(("SOURCE VOCAB SIZE", len(self.source_vocab)))
		print(("TARGET VOCAB SIZE", len(self.target_vocab)))

	def load_data(self, config):
		"""Load training data"""

		print(config)
		dataset = config["options"]["dataset"]
		df = load_piped_dataframe(dataset, chunksize=1000)

		for chunk in df:

			if len(chunk.columns) == 1:
				lines = chunk[chunk.columns[0]]
				text = " ".join(list(lines))
				thoughts = text.split(".")
			else: 
				thoughts = chunk["Thought"]

			for i, thought in enumerate(thoughts):
				if i + 1 < len(thoughts):
					self.source_lines.append(thoughts[i])
					self.target_lines.append(thoughts[i+1])

	def build_word_vocab(self):
		"""Build word vocab"""

		if os.path.isfile("models/wiki_word_lookup.json"):
			return load_json("models/wiki_word_lookup.json")

		corpus = self.source_lines + [self.target_lines[-1]]

		tknzr = TweetTokenizer().tokenize

		def tokenizer(thought):
			output = tknzr(thought)
			output = [o for o in output if len(o) > 2]
			return output

		halfpoint = int(len(corpus) / 2)
		half_corpus = corpus[:halfpoint]

		vectorizer = CountVectorizer(max_features=25000, tokenizer=tokenizer)
		count_vector = vectorizer.fit_transform(half_corpus).toarray()
		count_vector_2 = vectorizer.fit_transform(corpus[halfpoint:]).toarray()
		vocab = vectorizer.get_feature_names()

		# Merge word and character vocabs
		vocab += self.source_vocab.keys()
		word_count = len(vocab)
		index_lookup = dict(zip(vocab, range(word_count)))

		dict_2_json(index_lookup, "models/wiki_word_lookup.json")

		return index_lookup

	def word_indices_to_string(self, sentence, vocab):
		"""Collapse repeated word embedding indices to string
		and convert to string"""

		id_word = {vocab[token] : token for token in vocab}

		for i, group in groupby(sentence):
			print(group)
			print(i)

		for w in sentence:
			if id_word[w] == 'eol':
				break
			sent += id_word[w]

		return "".join(sent)