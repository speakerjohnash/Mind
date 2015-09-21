#!/usr/local/bin/python3.3

"""This module loads classifiers that run on the GPU

Created on Sep 21, 2015
@author: Matthew Sevrens
"""

import ctypes
import sys
import csv
import json

def load_label_map(filename):
	"""Load a permanent label map"""

	input_file = open(filename, encoding='utf-8')
	label_map = json.loads(input_file.read())
	input_file.close()

	return label_map

def get_CNN(model_name):
	"""Load a function to process thoughts using a CNN"""

	lualib = ctypes.CDLL("/home/ubuntu/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)

	# Must Load Lupa After the Preceding Line
	import lupa
	from lupa import LuaRuntime

	# Load Runtime and Lua Modules
	lua = LuaRuntime(unpack_returned_tuples=True)
	nn = lua.require('nn')
	model = lua.require('mind/lua/model')
	torch = lua.require('torch')
	cutorch = lua.require('cutorch')
	cunn = lua.require('cunn')

	# Load Config
	lua.execute('''
		dofile("mind/lua/config.lua")
	''')

	# Load CNN and Label map
	if model_name == "thought_type":
		label_map = {"1":"Reflect", "2":"State", "3": "Ask", "4":"Predict"}
		lua.execute('''
			model = Model:makeCleanSequential(torch.load("models/thought_type.t7b"))
		''')
	else:
		print("Requested CNN does not exist. Please reference an existing model")

	# Prepare CNN
	lua.execute('''
		model = model:type("torch.CudaTensor")
		cutorch.synchronize()

		alphabet = config.alphabet
		dict = {}
		for i = 1, #alphabet do
			dict[alphabet:sub(i,i)] = i
		end
	''')

	# Load Lua Functions
	lua.execute('''
		function stringToTensor (str, l, input)
			local s = str:lower()
			local l = l or #s
			local t = input or torch.Tensor(#alphabet, l)
			t:zero()
			for i = #s, math.max(#s - l + 1, 1), -1 do
				if dict[s:sub(i,i)] then
					t[dict[s:sub(i,i)]][#s - i + 1] = 1
				end
			end
			return t
		end
	''')

	make_batch = lua.eval('''
		function(trans)
			transLen = table.getn(trans)
			batch = torch.Tensor(transLen, #alphabet, 177)
			for k = 1, transLen do
				stringToTensor(trans[k], 177, batch:select(1, k))
			end
			return batch
		end
	''')

	list_to_table = lua.eval('''
		function(trans)
			local t, i = {}, 1
			for item in python.iter(trans) do
				t[i] = item
				i = i + 1
			end
			return t
		end
	''')

	process_batch = lua.eval('''
		function(batch)
			batchLen = batch:size(1)
			batch = batch:transpose(2, 3):contiguous():type("torch.CudaTensor")
			output = model:forward(batch)
			max, decision = output:double():max(2)
			labels = {}
			for k = 1, batchLen do
				labels[k] = decision:select(1, k)[1]
			end
			return labels
		end
	''')

	# Generate Helper Function
	def apply_CNN(trans, doc_key="description", label_key="CNN"):
		"""Apply CNN to thoughts"""
		
		trans_list = [' '.join(x[doc_key].split()) for x in trans]
		table_trans = list_to_table(trans_list)
		batch = make_batch(table_trans)
		labels = process_batch(batch)
		decisions = list(labels.values())
		
		for i, t in enumerate(trans):
			t[label_key] = label_map.get(str(decisions[i]), "")

		return trans

	return apply_CNN

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions; it should not be run from the console.")

