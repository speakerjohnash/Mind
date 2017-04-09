import os
from os import listdir
from os.path import isfile, join

import numpy as np

from nltk.corpus import comtrans

class TranslationData:
	def __init__(self, bucket_quant):

		als = comtrans.aligned_sents('alignment-en-fr.txt')

		# Load Aligned Translation Pairs
		self.source_lines = []
		self.target_lines = []

		for al in als:
			self.source_lines.append(' '.join(al.mots))
			self.target_lines.append(' '.join(al.words))

		print(("Source Sentences", len(self.source_lines)))
		print(("Target Sentences", len(self.target_lines)))

		self.bucket_quant = bucket_quant
		self.source_vocab = self.build_vocab(self.source_lines)
		self.target_vocab = self.build_vocab(self.target_lines)

		print(("SOURCE VOCAB SIZE", len(self.source_vocab)))
		print(("TARGET VOCAB SIZE", len(self.target_vocab)))

	def bucket_data(self):
		"""Bucket Data"""

		source_lines = []
		target_lines = []

		for i in range(len(self.source_lines)):
			source_lines.append( self.string_to_indices(self.source_lines[i], self.source_vocab) )
			target_lines.append( self.string_to_indices(self.target_lines[i], self.target_vocab) )

		buckets = self.create_buckets(source_lines, target_lines)

		frequent_keys = [ (-len(buckets[key]), key) for key in buckets ]
		frequent_keys.sort()

		print(("Source", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][0], self.source_vocab)))
		print(("Target", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][1], self.target_vocab)))
		
		print((len(frequent_keys)))
		return buckets, self.source_vocab, self.target_vocab, frequent_keys

	def create_buckets(self, source_lines, target_lines):
		"""Create Buckets"""
		
		bucket_quant = self.bucket_quant
		source_vocab = self.source_vocab
		target_vocab = self.target_vocab

		buckets = {}

		for i in range(len(source_lines)):
			
			source_lines[i] = np.concatenate( (source_lines[i], [source_vocab['eol']]) )
			target_lines[i] = np.concatenate( ([target_vocab['init']], target_lines[i], [target_vocab['eol']]) )
			
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

	def build_vocab(self, sentences):
		"""Build character vocab"""

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

		return vocab

	def string_to_indices(self, sentence, vocab):
		"""Convert string to embedding lookup indices"""

		indices = [vocab[s] for s in sentence]

		return indices

	def inidices_to_string(self, sentence, vocab):
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

class WikiData:
	def __init__(self, config):

		# Load Aligned Sequential Sentences
		self.source_lines = []
		self.target_lines = []

		reader = load_data(config)

	def load_data(config):
	"""Load training data"""

		dataset = config["options"]["dataset"]
		df = load_piped_dataframe(dataset, chunksize=1000)

		for chunk in df:

			if len(chunk.columns) == 1:
				lines = chunk[chunk.columns[0]]
				text = " ".join(list(lines))
				thoughts = text.split(".")
			else: 
				thoughts = chunk["Thought"]

		return df