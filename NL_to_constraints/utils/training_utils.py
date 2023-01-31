from collections import defaultdict

from .constants import *
from NL_constraints_data_prep.Utils.data_utils_helpers import decode_constraint
import torch
import torch.nn as nn
import csv


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

def init_output_file(conf_path, conf_name):
    file = conf_path + '/' + conf_name + '/' + 'test_outputs.csv'
    labels = ['', 'Map', 'Input Text', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4',
              'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8']
    with open(file, 'w') as myfile:
        csv_writer = csv.writer(myfile)
        csv_writer.writerow(labels)
    return file

def remove_count(l1: list):
    '''
    Remove count index from list of tuples
    '''
    return [l[0] for l in l1]

def compute_accuracy(predictions: torch.LongTensor, targets: torch.LongTensor):
    '''
    Compute number of correct predictions
    Returns number of correct predictions in a batch as well as total number of targets
    '''
    predictions = predictions.view(targets.shape)
    num_correct = 0
    num_total = 0
    for i in range(len(predictions)):
        l1 = predictions[i].tolist()
        l2 = targets[i].tolist()
        l1, l2 = convert_based_on_position(l1, l2)

        common = intersection(l1, l2)
        final = remove_count(common)
        num_correct += len(final)
        num_total += len(l2)
    return num_correct, num_total

def make_predictions(output_logits):
    predictions = output_logits.topk(k=1, dim=1)[1]
    return predictions

def invalid_constraint_loss(predictions):

    total = 0
    con_loss = 0
    for j in range(len(predictions)):
        pred_c, pred_v = decode_constraint(predictions[j].item())
        if pred_v in INVALID_VALUES[pred_c]:
            con_loss += 1
        total += 1

    return float(con_loss) / total, predictions


def epoch_time(start_time, end_time):
    '''
    Compute time taken per epoch in minutes+seconds
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        # else:
        #     nn.init.constant_(param.data, 0)

def check_early_stop(accs):
    '''
    Check if the past three accuracies have been the same. If so, stop
    '''
    if len(accs) > 3 and abs(accs[-1] - accs[-2]) < 0.00001 and abs(accs[-1] - accs[-3]) < 0.00001 and abs(accs[-1] - accs[-4]) < 0.00001 and abs(accs[-1] - accs[-5]) < 0.00001:
        return True
    else:
        return False