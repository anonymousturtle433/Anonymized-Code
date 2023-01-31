import torch
from collections import defaultdict
import math

def real_to_buckets(goals, range_limit=100, num_buckets=5):
    goals = torch.Tensor(goals)
    goals = torch.clamp(goals, -99.999, 99.999)
    # approximate to buckets (rounding)
    bucket_spacing = 2 * range_limit / num_buckets
    goals = torch.floor((goals + range_limit) / (
        bucket_spacing)).long()  # have to add the -1 to match indexing conventions
    return goals.detach()


def intersection(lst1: list, lst2: list):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def convert_based_on_position(l1: list, l2: list):
    '''
	Convert original list to a list of tuple with (item, count(item))
	[1,2,3,1] -> [(1,0), (2,0), (3,0), (1,1)]
	'''
    count = defaultdict(int)
    l1_new = []
    for i in range(len(l1)):
        l1_new.append((l1[i], count[l1[i]]))
        count[l1[i]] += 1
    count_l2 = defaultdict(int)
    l2_new = []
    for j in range(len(l2)):
        l2_new.append((l2[j], count_l2[l2[j]]))
        count_l2[l2[j]] += 1
    return l1_new, l2_new


def remove_count(l1: list):
    '''
	Remove count index from list of tuples
	'''
    return [l[0] for l in l1]


def compute_accuracy(predictions, targets):
    '''
	Compute number of correct predictions
	Returns number of correct predictions in a batch as well as total number of targets
	'''
    num_correct = 0
    num_total = 0
    l1 = predictions
    l2 = targets
    l1, l2 = convert_based_on_position(l1, l2)

    common = intersection(l1, l2)
    final = remove_count(common)
    num_correct += len(final)
    num_total += len(l2)

    return num_correct, num_total


def num_equals(l1, l2):
    num = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            num += 1
    return num
