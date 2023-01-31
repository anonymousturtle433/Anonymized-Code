import io
from tqdm import tqdm
import torch
import logging
from copy import deepcopy
from torchtext.vocab import Vocab
from collections import Counter
from enum import Enum
import numpy as np
from NL_to_constraints.utils.constants_no_empty_constraints import *
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

def json_iterator(data_path, yield_cls=False):
    '''
    Create iterator from json files consisting of goals and constraints.
    Load all the data from the json files and tokenize text
    :param data_path:
    :param yield_cls: Whether or not you need to use the iterator for classification
    :return:
    '''
    tokenizer = get_tokenizer("subword")
    with io.open(data_path, encoding="utf8") as f:
        data = json.load(f)
        for row in data:
            selections = row['Selections']
            maps = row['Map']
            goals = row['Goals']
            constraints = row['Constraints']
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

def create_data_from_iterator(vocab, iterator, include_unk):
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
        for selections, constraints, goals, maps, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                       for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((selections, constraints, goals, maps, tokens))
            labels.append((goals, constraints))
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

# def decode_tensor(tensor, vocab):
#     '''
#     Convert single tensor into language
#     '''
#     tokens = " ".join([vocab.itos[token] for token in tensor])
#     return tokens


def build_vocab_from_iterator(iterator):
    """
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
    word_vocab = Vocab(counter, min_freq=4, specials=['<pad>', '<unk>', '<sos>', '<eos>'])
    return word_vocab