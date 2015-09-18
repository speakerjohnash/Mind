import pandas as pd
import numpy as np

df = pd.read_csv("thought.csv", na_filter=False, encoding="utf-8", error_bad_lines=False)

grouped = df.groupby('Type', as_index=False)
groups = dict(list(grouped))
del groups['']
df = pd.concat(list(groups.values()), ignore_index=True)

label_map = dict(zip(groups.keys(), range(1, len(groups.keys()) + 1)))

a = lambda x: label_map[x["Type"]]
df['LABEL_NUM'] = df.apply(a, axis=1)

b = lambda x: " "
df['blank'] = df.apply(b, axis=1)

out = df[["LABEL_NUM", "blank", "Thought"]]

msk = np.random.rand(len(out)) < 0.90
train = out[msk]
test = out[~msk]

train.to_csv("train_thoughts.csv", header=False, index=False, index_label=False)
test.to_csv("test_thoughts.csv", header=False, index=False, index_label=False)