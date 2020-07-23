# %%

import numpy as np


a = np.array([2, 0, 1])

sorted(range(len(a)), key=lambda k: a[k])
# %%
import torch
from common.utils import load_data_with_pickle
PATH = 'resources/type_embeddings/choi_hyperbolic.pth'

e = torch.load(PATH)
embeddings_labels = e['objects']

# %%
import json
from tqdm import tqdm
with open('datasets/choi_dataset/choi_validation.json', 'r') as inp:
	lines = inp.readlines()

	examples = []
	for l in tqdm(lines):
		examples.append(json.loads(l))


# %%

found = []
not_present = []

for e in tqdm(examples):

	# if not flag and len(found) > len(embeddings_labels):
	# 	flag = True
	# 	big_list = found
	# 	small_list = embeddings_labels

	for label in e['y_str']:
		if label not in found and label not in not_present:
			if label in embeddings_labels:
				found.append(label)
			else:
				not_present.append(label)

#  %%

good_examples = []
bad_examples = []

for e in tqdm(examples):
	if any(x in not_present for x in e['y_str']):
		bad_examples.append(e)
	else:
		good_examples.append(e)


# %%

with open('datasets/choi_dataset/vimercati_datasets/validation_without_3h.json', 'w') as out:
	for e in tqdm(good_examples):
		json.dump(e, out)

# %%
