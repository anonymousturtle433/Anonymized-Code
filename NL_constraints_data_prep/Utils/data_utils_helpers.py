import io
from tqdm import tqdm
import torch
import logging
from copy import deepcopy
from torchtext.vocab import vocab as vocab_tt
import random

from collections import Counter
from enum import Enum
import numpy as np
import pandas as pd
from NL_to_constraints.utils.constants import *
from torchtext.data.utils import get_tokenizer

import json

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

constraint_types = {"I must have troops on ": 1, "I must not have troops on ": 2, "I must be able to access ": 3,
                        "I need to protect the borders of ": 4, "I need a total of at least ": 5,
                        "I must have at least ": 6, "I must have troops on at least ": 7, "I must place at least ": 8,
                        "I must have troops on at most ": 9}
value_names = {1: 'continent', 2: 'continent', 3: 'continent', 4: 'continent', 5: 'number', 6: 'number',
               7: 'number', 8: 'number', 9: 'number'}
value_types = {
    'continent': {'Blue': 1, 'Green': 2, 'Yellow': 3, 'Red': 4, 'Purple': 5},
    'number': {'1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, '10': 15, '11': 16,
               '12': 17, '13': 18, '14': 19},
    'num_countries': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6},
    'num_continents': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
}

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def test_duplicate_values(train_data, val_data):
    # Checking if there are overlaps between val and train data
    # We specifically check Selections, Goals and Constraints since we want to check overlaps between augmented points too.
    test = True
    for i, v in enumerate(val_data):
        for j, t in enumerate(train_data):
            if v['Selections'] == t['Selections'] and v['Goals'] == t['Goals'] and v['Constraints'] == t[
                'Constraints']:
                test = False
                break
        if test is False:
            break
    assert test == True, "There were values in the validation set that were already in the augmented training data. Please double check your dataset or code."

def populate_synthetic_data(syn_df):
    '''
    Fill in random values for the missing fields from the synthetic corpus
    :param syn_df: dataframe consisting of synthetic data
    :return: None
    '''
    syn_df['select_1'] = "Synthetic1"
    syn_df['select_2'] = "Synthetic2"
    syn_df['select_3'] = "Synthetic3"
    syn_df['select_4'] = "Synthetic4"
    syn_df['select_5'] = "Synthetic5"
    syn_df['select_6'] = "Synthetic6"
    syn_df['select_7'] = "Synthetic7"
    syn_df['quantity_1'] = "Synthetic1"
    syn_df['quantity_2'] = "Synthetic2"
    syn_df['quantity_3'] = "Synthetic3"
    syn_df['quantity_4'] = "Synthetic4"
    syn_df['quantity_5'] = "Synthetic5"
    syn_df['quantity_6'] = "Synthetic6"
    syn_df['quantity_7'] = "Synthetic7"
    syn_df['map'] = 999

def json_iterator(data_path, yield_cls=False, tokenizer = None):
    '''
    Create iterator from json files consisting of goals and constraints.
    Load all the data from the json files and tokenize text
    :param data_path:
    :param yield_cls: Whether or not you need to use the iterator for classification
    :return:
    '''
    if tokenizer == None:
        tokenizer = get_tokenizer("subword")
        tokenizer_type = 'torchtext'
    else:
        tokenizer = tokenizer
        tokenizer_type = 'huggingface'
    with io.open(data_path, encoding="utf8") as f:
        data = json.load(f)
        for row in data:
            selections = row['Selections']
            maps = row['Map']
            goals = row['Goals']
            constraints = row['Constraints']
            if tokenizer_type == 'huggingface':
                tokens = row['Text']
            else:
                tokens = tokenizer(row['Text'].lower().strip())
                tokens = [token.strip() for token in tokens if not token == '' and not token.isspace()]
                tokens = ['<sos>'] + tokens + ['<eos>']
                tokens = consolidate_territory_names(tokens)
            if yield_cls:
                yield selections, constraints, goals, maps, tokens
            else:
                yield tokens

def encode_constraint(con_class, value):
    """
    Convert constraint in the form of a (class, value) pair to a single number
    num = class * len(values) + value
    """
    return constraint_types[con_class] * len(VALUE_TYPES) + value_types[value_names[constraint_types[con_class]]][value]

def encode_constraint_no_empty(con_class, value):
    """
    Convert constraint in the form of a (class, value) pair to a single number
    num = class * len(values) + value
    """
    return (constraint_types[con_class] - 1) * len(VALUE_TYPES) + value_types[value_names[constraint_types[con_class]]][value] - 1


def consolidate_territory_names(tokens):
    '''
    Function to make sure that territory names are not tokenized, 'green_a', 'yellow_b', etc.
    :param tokens: list of original tokens
    :return: tokens: list of tokens with consolidated territory names
    '''
    colors = ['red', 'blue', 'green', 'purple', 'yellow']
    alpha = ['a', 'b', 'c', 'd', 'e']
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in colors:
            # Special case for dealing with territory names
            # Tokens like 'green_a' or 'yellow_b' should be
            # considered as individual tokens rather than 'green' + 'a'
            if i < len(tokens) - 2:
                if tokens[i + 1] == '_' and tokens[i + 2] in alpha:
                    # Check if a territory name was tokenized as 'green' + '_' + 'a'
                    new_token = ''.join(tokens[i:i + 3])
                    new_tokens.append(new_token)
                    i += 3
                elif tokens[i + 1] in alpha:
                    # Check if a territory name was tokenized as 'green' + 'a'
                    new_token = '_'.join(tokens[i:i + 2])
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            else:
                # If the number of tokens left in the sentence is less than
                new_tokens.append(tokens[i])
                i += 1
        else:
            new_tokens.append(tokens[i])
            i += 1
    tokens = deepcopy(new_tokens)
    return tokens

def create_data_from_iterator(vocab, iterator, include_unk, tokenizer = None):
    '''
    Convert Json file into dataset with text and G+C labels
    :param vocab:
    :param iterator:
    :param include_unk:
    :return: data: str, lanels: (goals: list, constraints: list)
    '''
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        count = 0
        for selections, constraints, goals, maps, tokens in iterator:
            if tokenizer != None:
                tokens = tokens
            elif include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not vocab['<unk>'], [vocab[token]
                                                                       for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((selections, constraints, goals, maps, tokens))
            labels.append((goals, constraints))
            count += 1
            t.update(1)
    return data, labels

def decode_constraint(num):
    '''
    Convert encoded constraint back into (class, value) tuple
    '''
    return (int(num / len(VALUE_TYPES)), num % len(VALUE_TYPES))

def decode_batch(batch, vocab):
    '''
    Decode tensor of inputs back into text
    :param tensor: encoded inputs
    :return: text: str
    '''
    tokens = [decode_tensor(tensor, vocab) for tensor in torch.transpose(batch.clone(),0,1)]
    return tokens

def decode_tensor(tensor, vocab):
    '''
    Convert single tensor into language
    '''
    tokens = " ".join([vocab.get_itos()[token] for token in tensor])
    return tokens


def build_vocab_from_iterator(iterator):
    """
    DEPRECATED WITH TORCHTEXT==0.10.0
    Build a Vocab from an iterator.
    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    """

    counter = Counter()
    colors = ['red', 'blue', 'green', 'purple', 'yellow']
    alpha = ['a', 'b', 'c', 'd', 'e']
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = vocab_tt(counter, min_freq=4, specials=['<pad>', '<unk>', '<sos>', '<eos>'])
    return word_vocab

def convert_to_text_goals(batch, tokenizer=None, vocab=None):
  '''
  Convert a batch of outputs to text. Set up for use with the goal model
  Args:
      batch: batch of data encoded with the current tokenizer
      tokenizer: tokenizer used to encode the batch
      vocab: used for the rnn decoder, unused for transformer/bert decoders

  Returns: text = list[str] -> List of decoded strings
  '''
  if type(tokenizer) == ByteLevelBPETokenizer: #transformer
    text = []
    for i in range(batch['input_ids'].size()[0]):
      text.append(tokenizer.decode(list(batch['input_ids'][i])))
  elif tokenizer == None: # rnn
    text = decode_batch(batch, vocab)
  else: #bert
    text = tokenizer.batch_decode(batch['input_ids'])
  return text

def read_data(filename, output_dir):
    df = pd.ExcelFile(output_dir + '/' + filename).parse('Sheet1')

    data = []
    count = 0
    # print(df.shape)
    for j in range(df.shape[0]):
        goals = []
        text = df.iloc[j]['Instructions']
        if text == "" or not text == text:
            continue
        for i in range(1, 7):
            goal_num = 'G' + str(i) + '_1'
            goals.append(int(df.iloc[j][goal_num]))
        selections = []
        selections.append(('<sos>', '<sos>'))
        empty_flag = False
        for i in range(1, 8):
            select_num = 'select_' + str(i)
            if isinstance(df.iloc[j][select_num], float):
                if j == 0:
                    # If the first selection is empty, set flag to true and break out of the loop
                    empty_flag = True
                break
                # break
            if "Synthetic" in df.iloc[j][select_num]:
                selections.append(('<pad>', '<pad>'))
                continue
            if 'select' not in df.iloc[j][select_num]:
                selections.append((df.iloc[j][select_num], int(df.iloc[j]['quantity_' + str(i)])))
        selections.append(('<eos>', '<eos>'))
        for l in range(len(selections), 9):
            selections.append(('<pad>', '<pad>'))
        if empty_flag:
            # Don't include data if all selections are empty
            continue
        constraints = []
        for i in range(1, 9):
            constraint_num = 'constraint_' + str(i)
            constraint_text = df.iloc[j][constraint_num]
            if type(constraint_text) == float:
                continue
            max_len = -100
            max_ty = None
            for ty in constraint_types.keys():
                if isinstance(constraint_text, float):
                    break
                if constraint_text.startswith(ty):
                    if len(ty.split()) > max_len:
                        max_len = max(len(ty.split()), max_len)
                        max_ty = ty
            ty = max_ty
            if max_ty:
                value = constraint_text.split(ty)[1].split(' ')[0]
                # constraints.append(
                #     (constraint_types[ty], value_types[value_names[constraint_types[ty]]][value]))
                # Convert constraint pair into a single digit
                # constraints.append(constraint_types[ty] * len(VALUE_TYPES) + value_types[value_names[constraint_types[ty]]][value])
                constraints.append(encode_constraint(ty, value))
        for l in range(len(constraints), 8):
            # constraints.append((0, 0))
            constraints.append(0)
        data_dict = {'Map': df.iloc[j]['map'], 'Selections': selections, 'Text': text, 'Goals': goals,
                     'Constraints': constraints}
        data.append(data_dict)
    random.shuffle(data)
    return data