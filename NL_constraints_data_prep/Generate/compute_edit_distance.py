import edit_distance
import csv
from tqdm import tqdm
import numpy as np


# ref = [1, 2, 5, 4, 3, 6]
# hyp = [1, 2, 3, 4, 5, 6]
# sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
# sm.get_opcodes()
# sm.ratio()
# sm.get_matching_blocks()
# sm.distance()

def prep_and_tokenize(text):
  tokens = [token for token in text.lower().strip().split(" ") if token != ' ']
  return tokens

with open("../Data/Augmented_To_Synthetic_Strategy_Comparison.csv") as f:
  reader = csv.reader(f)
  data = []
  for row in reader:
    if len(row) > 0:
      data.append(row[0])

data = data[1:] #throw out row with headers
paired_data = []
for i in range(len(data)//2):
  paired_data.append((data[i*2], data[i*2 + 1]))

dists = []
orig_lens = []
aug_lens = []
for orig, aug in tqdm(paired_data):
  orig_tok = prep_and_tokenize(orig)
  aug_tok = prep_and_tokenize(aug)
  orig_lens.append(len(orig_tok))
  aug_lens.append(len(aug_tok))


  sm = edit_distance.SequenceMatcher(a=orig_tok, b=aug_tok)

  dists.append(sm.distance())

np_dists = np.array(dists)
np_orig_lens = np.array(orig_lens)
np_aug_lens = np.array(aug_lens)

print('Edit distance Stats:')
print("Mean: ", np_dists.mean())
print("Max: ", np_dists.max())
print("Min: ", np_dists.min())

print('Original Length Stats:')
print("Mean: ", np_orig_lens.mean())
print("Max: ", np_orig_lens.max())
print("Min: ", np_orig_lens.min())

print('Augmented Length Stats:')
print("Mean: ", np_aug_lens.mean())
print("Max: ", np_aug_lens.max())
print("Min: ", np_aug_lens.min())