import csv
import sys
import math

import pandas as pd
import numpy as np

from mind.load_CNN import get_CNN 

#################### USAGE ##########################

# python3.3 -m mind.apply_CNN [file_name] [model_name] [column_name]
# python3.3 -m mind.apply_CNN data/input/thought.csv thought_type Thought_Type_CNN

#####################################################

CLASSIFIER = get_CNN(sys.argv[2])

df = pd.read_csv(sys.argv[1], na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep="|", error_bad_lines=False)
trans = list(df.T.to_dict().values())
trans = CLASSIFIER(trans, doc_key="Thought", label_key=sys.argv[3])
out_df = pd.DataFrame(trans)
out_df.to_csv("data/output/classified_thought.csv", sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
