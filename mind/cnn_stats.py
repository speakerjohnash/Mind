#!/usr/local/bin/python3.3

"""This module processes the output of Crepe into a more
readable form

Created on Jan 15, 2016
@author: Matthew Sevrens
"""

import ctypes
import json
import sys
import pandas as pd
import numpy as np

#################### USAGE ##########################

# Note: In Progress
# python3 -m mind.cnn_stats [main_filename] [label_map]
# python3 -m mind.cnn_stats path/to/main_10000.t7b path/to/label_map.json

#####################################################

lualib = ctypes.CDLL("/home/ubuntu/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)

# Must be imported after previous statement
import lupa
from lupa import LuaRuntime

# Load Runtime and Lua Modules
lua = LuaRuntime(unpack_returned_tuples=True)
torch = lua.require('torch')

# Load Lua Functions
load_cm = lua.eval('''
	function(filename)
		rows = {}
		main = torch.load(filename)
		cm = main.record[#main.record]['val_confusion']
		for i = 1, cm:size(1), 1 do
			column = {}
			for k = 1, cm:size(2), 1 do
				column[k] = cm:select(1, i)[k]
			end
			rows[i] = column
		end
		return rows
	end
''')

cm = load_cm(sys.argv[1])
size = len(cm)
cm2 = []

for i in range(1, size+1):
	row = []
	for j in range(1, size+1):
		row.append(cm[i][j])
	cm2.append(row)

true_positive = pd.DataFrame([cm2[i][i] for i in range(size)])
cm2 = pd.DataFrame(cm2)
actual = pd.DataFrame(cm2.sum(axis=1))
recall = true_positive / actual
recall = np.round(recall, decimals=3)
column_sum = pd.DataFrame(cm2.sum())
false_positive = column_sum - true_positive
precision = true_positive / column_sum
precision = np.round(precision, decimals=3)
misclassification = actual - true_positive
label = pd.DataFrame(pd.read_json(sys.argv[2], typ='series')).sort_index()
label.index = range(size)

stat = pd.concat([label, actual, true_positive, false_positive, recall, precision, misclassification], axis=1)
stat.columns = ['Class', 'Actual', 'True_Positive', 'False_Positive', 'Recall', 'Precision', 'Misclassification']

cm2 = pd.concat([label, cm2], axis=1)
cm2.columns = ['Class'] + [str(x) for x in range(size)]

stat.to_csv('CNN_stat.csv', index=False)
cm2.to_csv('Con_Matrix.csv')
