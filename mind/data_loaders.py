import sys
import os
from os import listdir
from os.path import isfile, join
from itertools import groupby

import numpy as np

from nltk.corpus import comtrans
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from mind.tools import load_piped_dataframe, dict_2_json, load_json, load_dict_list

class TranslationData():
	def __init__(self, bucket_quant, config):

		self.config = config
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
		self.source_vocab = self.build_char_vocab(self.source_lines, "source")
		self.target_char_vocab = self.build_char_vocab(self.target_lines, "target")
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

	def build_char_vocab(self, sentences, name):
		"""Build character vocab"""

		if "resume_model" in self.config and os.path.isfile("models/" + name + "_char_lookup.json"):
			print("Restoring previous character lookup")
			return load_json("models/" + name + "_char_lookup.json")

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

		dict_2_json(vocab, "models/" + name + "_char_lookup.json")

		return vocab

	def build_word_vocab(self):
		"""Build word vocab"""

		if "resume_model" in self.config and os.path.isfile("models/word_lookup.json"):
			print("Restoring previous word lookup")
			return load_json("models/word_lookup.json")

		tknzr = TweetTokenizer().tokenize

		def tokenizer(thought):
			output = tknzr(thought)
			output = [o for o in output if len(o) > 2]
			return output

		corpus = self.target_lines

		vectorizer = CountVectorizer(max_features=25000, max_df=0.98, tokenizer=tokenizer)
		count_vector = vectorizer.fit_transform(corpus).toarray()
		feature_names = list(vectorizer.get_feature_names())
		feature_names.remove("the")
		feature_names.remove("and")

		# Merge word and character vocabs
		vocab = list(self.target_char_vocab.keys())
		vocab += feature_names
		word_count = len(vocab)
		index_lookup = dict(zip(vocab, range(word_count)))

		dict_2_json(index_lookup, "models/word_lookup.json")

		return index_lookup

	def string_to_word_indices(self, sentence, vocab):
		"""Convert string to repeated word embedding indices
		equal the length of each word"""

		tokens = sentence.split(" ")
		indices = []

		for i, token in enumerate(tokens):
			if token in vocab:
				for ii in range(len(token)): 
					indices.append(vocab[token])
			else:
				for char in token:
					indices.append(vocab.get(char, vocab[" "]))

			if i != len(tokens):
				indices.append(vocab[" "])

		return indices

	def word_indices_to_string(self, sentence, vocab):
		"""Collapse repeated word embedding indices to string
		and convert to string"""

		id_word = {vocab[token] : token for token in vocab}
		sent = []

		for i, group in groupby(sentence):

			if id_word[i] == 'eol':
				break

			if i in self.target_char_vocab.values():
				for g in list(group):
					sent += id_word[i]
			else:
				sent += id_word[i]

		return "".join(sent)

	def string_to_char_indices(self, sentence, vocab):
		"""Convert string to embedding lookup indices"""

		indices = [vocab.get(s, vocab[" "]) for s in sentence]

		return indices

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

class PretrainData(TranslationData):
	def __init__(self, bucket_quant, config):

		self.config = config
		self.options = config["prophet"]

		# Load Aligned Sequential Sentences
		self.source_lines = []
		self.target_lines = []

		self.load_dictionary()
		self.load_data("data/Nate_Silver_The_Signal_and_the_Noise.txt")
		self.load_data(config["options"]["dataset"])

		print(("Source Sentences", len(self.source_lines)))
		print(("Target Sentences", len(self.target_lines)))

		# Build word and character vocabs
		self.bucket_quant = config["options"]["bucket_quant"]
		self.source_vocab = self.build_char_vocab(self.source_lines, "source")
		self.target_char_vocab = self.build_char_vocab(self.target_lines, "target")
		self.target_vocab = self.build_word_vocab()

		print(("SOURCE VOCAB SIZE", len(self.source_vocab)))
		print(("TARGET VOCAB SIZE", len(self.target_vocab)))

	def load_dictionary(self):
		"""Load a dictionary as the corpus"""

		thoughts = load_dict_list("data/prophet_lexicon.csv")
		corpus = [t["Thought"] for t in thoughts]

		for item in corpus:

			item = item.replace("\t","")
			split = item.split(": ")

			if len(split) != 2:
				continue

			word = split[0]
			definition = split[1]

			self.source_lines.append(word[:254])
			self.target_lines.append(word[:254])

			self.source_lines.append(definition[:254])
			self.target_lines.append(definition[:254])
		
		# dictionary = load_json("data/dictionary.json")

		# for key, value in dictionary.items():
		#	self.source_lines.append(key)
		#	self.target_lines.append(value)

	def load_data(self, dataset):
		"""Load training data"""

		df = load_piped_dataframe(dataset, chunksize=1000)

		for chunk in df:

			if len(chunk.columns) == 1:
				lines = chunk[chunk.columns[0]]
				text = " ".join(list(lines))
				thoughts = text.split(".")
			else: 
				thoughts = chunk["Thought"]

			# TODO Split sentences longer than sample size into multiple sentences

			for i, thought in enumerate(thoughts):
				if i + 1 < len(thoughts):
					thought = thoughts[i][:254]
					if len(thought) < 50:
						continue
					self.source_lines.append(thought)
					self.target_lines.append(thought)

	def build_word_vocab(self):
		"""Build target vocab"""

		if "resume_model" in self.config and os.path.isfile("models/pretrain_word_lookup.json"):
			print("Loading previous word lookup")
			return load_json("models/pretrain_word_lookup.json")

		thoughts = load_dict_list("data/thoughts.csv")
		corpus = [t["Thought"] for t in thoughts]

		# corpus = self.source_lines + [self.target_lines[-1]]

		tknzr = TweetTokenizer().tokenize

		def tokenizer(thought):
			output = tknzr(thought)
			output = [o for o in output if len(o) > 2]
			return output

		third = int(len(corpus) / 3)
		first_section = corpus[:third]
		second_section = corpus[third:third+third]
		third_section = corpus[third+third:]

		vectorizer = CountVectorizer(max_features=25000, tokenizer=tokenizer)

		vectorizer.fit_transform(first_section)
		vectorizer.fit_transform(second_section)
		vectorizer.fit_transform(third_section)

		feature_names = list(vectorizer.get_feature_names())
		feature_names.remove("the")
		feature_names.remove("and")

		# Merge word and character vocabs
		vocab = list(self.target_char_vocab.keys())
		vocab += feature_names
		word_count = len(vocab)
		index_lookup = dict(zip(vocab, range(word_count)))

		dict_2_json(index_lookup, "models/pretrain_word_lookup.json")

		return index_lookup

	def build_char_vocab(self, sentences, name):
		"""Build source vocab"""

		if "resume_model" in self.config and os.path.isfile("models/" + name + "_pretrain_char_lookup.json"):
			print("Loading previous character lookup")
			return load_json("models/" + name + "_pretrain_char_lookup.json")

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

		dict_2_json(vocab, "models/" + name + "_pretrain_char_lookup.json")

		return vocab

	def load_batch(self, step, buckets):
		"""Load a batch of documents"""

		options = self.options
		sample_size = options["sample_size"]
		batch_size = options["batch_size"]

		source_sentences = []
		target_sentences = []

		sentences = buckets[sample_size][step : step + batch_size]

		for s, t in sentences:
			source_sentences.append(s)
			target_sentences.append(t)
		
		source_sentences = source_sentences[:batch_size - 1]
		target_sentence = target_sentences[len(target_sentences) - 1]

		# Vary target by step
		# if step % 2 == 0:
			# Predict Next Thought
		#	target_sentence = target_sentences[len(target_sentences) - 1]
		#else:
			# Recall Last Thought
		#	target_sentence = target_sentences[len(target_sentences) - 2] 

		return np.array(source_sentences, dtype = 'int32'), np.array([target_sentence], dtype = 'int32')

	def create_buckets(self, source_lines, target_lines):
		"""Create buckets"""

		options = self.options
		sample_size = options["sample_size"]

		bucket_quant = self.bucket_quant
		source_vocab = self.source_vocab
		target_vocab = self.target_vocab

		buckets = {}

		for i in range(len(source_lines)):
			
			source_lines[i] = np.concatenate((source_lines[i], [source_vocab['eol']]))
			target_lines[i] = np.concatenate(([target_vocab['init']], target_lines[i], [target_vocab['eol']]))
			
			sl = len(source_lines[i])
			tl = len(target_lines[i])

			new_length = sample_size
			
			s_padding = np.array([source_vocab['padding'] for ctr in range(int(sl), int(new_length))])

			# Extra Padding for Training
			t_padding = np.array([target_vocab['padding'] for ctr in range(int(tl), int(new_length + 1))])
			source_lines[i] = np.concatenate([source_lines[i], s_padding])
			target_lines[i] = np.concatenate([target_lines[i], t_padding])

			if sample_size in buckets:
				buckets[new_length].append((source_lines[i], target_lines[i]))
			else:
				buckets[sample_size] = [(source_lines[i], target_lines[i])]

			if i%1000 == 0:
				print(("Loading", i))
			
		return buckets

	def bucket_data(self):
		"""Bucket Data"""

		options = self.options
		sample_size = options["sample_size"]
		source_lines = []
		target_lines = []

		for i in range(len(self.source_lines)):
			source_lines.append(self.string_to_char_indices(self.source_lines[i], self.source_vocab))
			target_lines.append(self.string_to_word_indices(self.target_lines[i], self.target_vocab))

		buckets = self.create_buckets(source_lines, target_lines)

		frequent_keys = [(-len(buckets[key]), key) for key in buckets]
		frequent_keys.sort()

		print(("Source", self.char_indices_to_string(buckets[sample_size][5][0], self.source_vocab)))
		print(("Target", self.word_indices_to_string(buckets[sample_size][5][1], self.target_vocab)))
		
		print((len(frequent_keys)))

		return buckets, self.source_vocab, self.target_vocab, frequent_keys

if __name__ == "__main__":

	# Test Encoding of Mixed Embeddings for Targets
	config = load_json("config/mind_config.json")
	dl = TranslationData(25, config)
	corpus = dl.target_lines
	encoded = dl.string_to_word_indices(corpus[0], dl.target_vocab)
	decoded = dl.word_indices_to_string(encoded, dl.target_vocab)

	print(corpus[0])
	print(encoded)
	print(decoded)