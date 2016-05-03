import csv
import sys
import math

from itertools import zip_longest

import pandas as pd
import numpy as np

from mind.load_CNN import get_CNN 

#################### USAGE ##########################

# python3.3 -m mind.apply_cnn [file_name] [model_name] [column_name]
# python3.3 -m mind.apply_cnn data/input/thought.csv thought_type Thought_Type_CNN

#####################################################

def grouper(iterable):
	return zip_longest(*[iter(iterable)]*1000, fillvalue={"Thought":""})

CLASSIFIER = get_CNN(sys.argv[2])

df = pd.read_csv(sys.argv[1], na_filter=False, encoding="utf-8", error_bad_lines=False)
thoughts = list(df.T.to_dict().values())

processed = []
batches = grouper(thoughts)
	
for i, batch in enumerate(batches):
	processed += CLASSIFIER(batch, doc_key="Thought", label_key=sys.argv[3])

processed = processed[0:len(thoughts)]
out_df = pd.DataFrame(processed)
out_df.to_csv("data/output/classified_thought.csv", sep="|", mode="w", encoding="utf-8", index=False, index_label=False)