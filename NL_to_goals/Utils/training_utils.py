import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
from transformers import get_linear_schedule_with_warmup
# from torchtext.data import Field, BucketIterator, Iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import time
import math
import itertools
import numpy as np
import itertools
from copy import deepcopy

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, RobertaConfig, RobertaTokenizer, RobertaModel
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from NL_to_goals.models.test_bert_to_goal import GoalLabelSmoothingLoss
from NL_to_constraints.utils.constants import *
from NL_to_goals.Utils.map_structures import *
from NL_to_goals.Utils.constants import NON_UNIFORM_BUCKETS

def initialize_variables(input_vocab):
    global vocab
    vocab = input_vocab

def collate_fn(batch, tokenizer, selection_type=None, goal_tokens=False):
  """
  collate function for rnn dataloaders
  :param batch: portion of data from dataset. See setup_datasets for more details on data from
  :param tokenizer: the tokenizer to apply to the raw text
  :param selection_type: the type of selection encoding that is being used
  :return: batch information in a form that can be processed by the training loop. This basically means that the
  text, constraints, goals, maps, and troop selection have been split up into individual lists or tensors
  """
  texts, constraints, selections, goals, maps = [], [], [], [], []
  for selection, constraint, goal, map, txt in batch:
    ignore_selections = ['<sos>', '<eos>','<pad>']
    continent_dict = {'Yellow': 0, 'Green': 1, 'Red': 2, 'Purple': 3, 'Blue': 4}
    selection_token = []
    if selection_type == 'full_one_hot_no_map':
      for s in selection:
        if s[0] not in ignore_selections:
          country_selection = np.zeros((5,1))
          continent_selection = np.zeros((5, 1))
          troops = np.zeros((14, 1))
          continent, country = s[0].split('_')
          continent_selection[continent_dict[continent]] = 1
          country_selection[ord(country) - ord('A')] =1
          troops[s[1]-1] = 1
          selection_token.append(np.concatenate([continent_selection,country_selection, troops]))
        elif s[0] == '<pad>':
          selection_token.append(np.zeros((24,1)))
    elif selection_type == 'full_one_hot_with_map' :
      for player in [selection, map_player_1[map - 1], map_player_2[map - 1]]:
        for s in player:
          if s[0] not in ignore_selections:
            country_selection = np.zeros((5,1))
            continent_selection = np.zeros((5, 1))
            troops = np.zeros((14, 1))
            continent, country = s[0].split('_')
            continent_selection[continent_dict[continent]] = 1
            country_selection[ord(country) - ord('A')] =1
            troops[s[1]-1] = 1
            selection_token.append(np.concatenate([continent_selection,country_selection, troops]))
          elif s[0] == '<pad>':
            selection_token.append(np.zeros((24,1)))
    elif selection_type == 'partial_one_hot_with_troops_no_map':
      selection_token = np.zeros((21, 1))
      for s in selection:
        if s[0] not in ignore_selections:
          selection_token[country_map[s[0]]-3] = s[1]
    elif selection_type == 'partial_one_hot_with_troops_with_map' :
      selection_token = np.zeros((63, 1))
      for player in [selection, map_player_1[map - 1], map_player_2[map - 1]]:
        for s in player:
          if s[0] not in ignore_selections:
            selection_token[country_map[s[0]]-3] = s[1]
    elif selection_type == 'text_selections_no_map': #selection encoding directly in text with no map
      selection_text = '[SEL]:'
      for k in range(len(selection)):
        if '<eos>' not in selection[k] and '<pad>' not in selection[k] and '<sos>' not in selection[k]:
          selection_text += selection[k][0] + ' = ' + str(selection[k][1]) + '|'
      selection_text = selection_text[:-1]
      selection_text += '[/SEL]'
      txt += ' ' + selection_text

      selection_var = deepcopy(selection)
      for s in selection_var:
        if type(s[0]) == str:
          s[0] = COUNTRY_MAP[s[0]]
          if s[0] < 3:
            s[1] = COUNTRY_MAP[s[1]]
      
      selections.append(torch.tensor(selection_var))
    elif selection_type == 'text_selections_map': #selection encoding directly in text with a map
      start_tokens = ['[SEL]: ', '[1SEL]:', '[2SEL]:']
      end_tokens   = ['[/SEL]', '[/1SEL]', '[/2SEL]']
      player_selections = [selection, map_player_1[map - 1], map_player_2[map - 1]]

      txt += ' '
      for start_token, end_token, current_selections in zip(start_tokens, end_tokens, player_selections):
        selection_text = start_token
        for k in range(len(current_selections)):
          if '<eos>' not in current_selections[k] and '<pad>' not in current_selections[k] and '<sos>' not in current_selections[k]:
            selection_text += current_selections[k][0] + ' = ' + str(current_selections[k][1]) + '|'
        selection_text = selection_text[:-1]
        selection_text += end_token
        txt += selection_text

      selection_var = deepcopy(selection)
      for s in selection_var:
        if type(s[0]) == str:
          s[0] = COUNTRY_MAP[s[0]]
          if s[0] < 3:
            s[1] = COUNTRY_MAP[s[1]]
      
      selections.append(torch.tensor(selection_var))
    elif selection_type == 'text_selections_simple_map':  # selection encoding directly in text with no map
      selection_text = '[SEL]:'
      for k in range(len(selection)):
        if '<eos>' not in selection[k] and '<pad>' not in selection[k] and '<sos>' not in selection[k]:
          selection_text += selection[k][0] + ' = ' + str(selection[k][1]) + '|'
      selection_text = selection_text[:-1]
      selection_text += '[/SEL]'
      txt += ' ' + selection_text
      txt += ' ' + f'[Map = {map}]' #note that maps are indexed from 1

      selection_var = deepcopy(selection)
      for s in selection_var:
        if type(s[0]) == str:
          s[0] = COUNTRY_MAP[s[0]]
          if s[0] < 3:
            s[1] = COUNTRY_MAP[s[1]]
      
      selections.append(torch.tensor(selection_var))

    if goal_tokens: #if using individual goal tokens for predictions
      txt = 'G1G2G3G4G5G6' + txt

    texts.append(txt)

    if selection_type != 'text_selections_no_map' and selection_type != 'text_selections_map' and selection_type != 'text_selections_simple_map':
      selections.append(torch.tensor(selection_token))
    constraint = torch.tensor(constraint)
    constraints.append(constraint)
    goals.append(torch.tensor(goal))
    maps.append(map)

  if tokenizer == None: #corresponds to rnn
    texts = pad_sequence(texts, batch_first = False, padding_value=vocab.get_stoi()['<pad>'])
  else:
    texts_input_ids = []
    texts_attention = []
    if type(tokenizer) == ByteLevelBPETokenizer:  # transformer
        text_res = tokenizer.encode_batch(texts)
        for res in text_res:
          texts_input_ids.append(torch.tensor(res.ids))
          texts_attention.append(torch.tensor(res.attention_mask))
    else: #bert
      for text in texts:
        tokens = tokenizer(text, return_tensors='pt')
        texts_input_ids.append(tokens['input_ids'].squeeze())
        texts_attention.append(tokens['attention_mask'].squeeze())
      # roberta padding scheme
    texts_input_ids = pad_sequence(texts_input_ids, batch_first=False, padding_value=1).T
    texts_attention = pad_sequence(texts_attention, batch_first=False, padding_value=0).T
    texts = BatchEncoding({'input_ids': texts_input_ids, 'attention_mask': texts_attention})
  goals = torch.stack(goals)
  maps = torch.tensor(maps)
  constraints = torch.stack(constraints)
  selections = torch.stack(selections)
  return texts, constraints, goals, maps, selections

def real_to_bucket_range(goals:Tensor, range_limit=100, num_buckets=5):
  """
  This function is used to transform real value goals to real valued goals within a limited range, which can then be compared to regression targets directly
  :param goals: tensor containing goal values
  :param range_limit: positive and negative range that goals can lie within; defaults to 100. Example: range_limit=100 -> goals in range -100 to 100
  :param num_buckets: number of buckets to map the raw goal values to; defaults to 5
  :return: tensor with the corresponding goal bucket for each initial goal value
  """
  goals = goals/(torch.max(torch.abs(goals)))
  bucket_spacing = 2*range_limit/num_buckets
  goals = (goals*range_limit + range_limit)/(bucket_spacing)
  return goals

def real_to_buckets(goals, range_limit=100, num_buckets=5, uniform_buckets=True, device='cpu'):
  """
  This function is used to transform real value goals within a limited range to buckets, which can then be used as classification targets
  :param goals: tensor containing goal values
  :param range_limit: positive and negative range that goals can lie within; defaults to 100. Example: range_limit=100 -> goals in range -100 to 100
  :param num_buckets: number of buckets to map the raw goal values to; defaults to 5
  :return: tensor with the corresponding goal bucket for each initial goal value
  """
  #note that this function is not differentiable due to rounding
  #using floor instead of round and + a small perturbation to ensure that the maximum value is assigned the correct index

  if uniform_buckets:
    goals = torch.clamp(goals, -99.999, 99.999)
    # approximate to buckets (rounding)
    bucket_spacing = 2 * range_limit / num_buckets
    goals = torch.floor((goals + range_limit) / (bucket_spacing)).long()
  else:
    #assumes that there are 6 goals currently (based on number of ranges in the NON_UNIFORM_BUCKETS constant)
    bucket_goals = -1 * torch.ones(goals.size()).long().to(device)
    goal_ranges = NON_UNIFORM_BUCKETS[num_buckets]
    for goal_num, current_range in enumerate(goal_ranges):
      for i in range(num_buckets):
        l, u = torch.tensor(current_range[i]).long().to(device), torch.tensor(current_range[i + 1]).long().to(device)
        bucket_goals[:, goal_num] = torch.where(torch.logical_and(goals[:, goal_num] >= l, goals[:, goal_num] < u),  torch.tensor(i).long().to(device), bucket_goals[:, goal_num])
    bucket_goals = torch.where(goals >= 100, torch.tensor(num_buckets - 1).long().to(device), bucket_goals)
    goals = bucket_goals
  return goals

def init_weights(m):
  """
  Given model, initialize parameters randomly
  :param m: model
  :return: None
  """
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module):
  """
  Counts the number of parameters in a pytorch model
  :param model: pytorch model
  :return: number of parameters
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
  """
  Calculates the amount of time each training epoch takes to run
  :param start_time: start time in seconds
  :param end_time:  end time in seconds
  :return: minutes elapsed, seconds elapsed (not including minutes)
  """
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def train(model: nn.Module,
          iterator,
          optimizer: optim.Optimizer,
          device,
          range: int,
          num_buckets: int,
          clip: float,
          criterion,
          scheduler=None,
          uniform_buckets=True,
          mse_weight=0.5):
    """
    Train the model on the provided data for one epoch
    :param model: pytorch model
    :param iterator: data to train on
    :param optimizer: optimizer being used for model training
    :param device: device which model and data reside on
    :param range: upper/lower limit within which the goal values fall
    :param num_buckets: number of buckets to train/evaluate on
    :param clip: gradient clipping parameter
    :param criterion: pytorch criterion object (either MSE or cross entropy)
    :param scheduler: pytorch scheduler object, used to asjust the learning rate throughout training
    :param uniform_buckets: where to use uniform buckets. If set to false, will use bucket ranges defined in NL_to_goals.Utils.constants
    :param mse_weight: when using combined CE and MSE loss, can be used to change weight on mse loss value relative to CE loss. Default is 50/50
    :return: the models loss for this epoch
    """

    model.train()

    epoch_loss = 0

    for batch in iterator:
      text = batch[0].to(device)
      selections = batch[4].to(device)
      goals = batch[2].to(device).float()
      bucket_goals = real_to_buckets(goals, range, num_buckets, uniform_buckets=uniform_buckets, device=device).to(device)

      optimizer.zero_grad()

      output = model(text, selections)

      if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, GoalLabelSmoothingLoss):
        loss = criterion(output.permute([0, 2, 1]), bucket_goals.long())  # crossentropy expects the class dimension to be in position 1
      elif isinstance(criterion, nn.L1Loss) or isinstance(criterion,
                                                          nn.MSELoss):  # only works with a regression network, otherwise it makes no sense
        bucket_range_outputs = real_to_bucket_range(output, range, num_buckets)
        loss = criterion(bucket_range_outputs, bucket_goals.float())
      elif type(criterion) == tuple:  # criterion in the form (CrossEntropy, Distance loss), only works with classification network
        cross_entropy_loss = criterion[0](output.permute([0, 2, 1]), bucket_goals.long())
        distance_loss = criterion[1](torch.argmax(output, dim=2).float(), bucket_goals.float())
        ce_weight = 1 - mse_weight
        loss = ce_weight*cross_entropy_loss + mse_weight*distance_loss
      elif isinstance(criterion, nn.BCELoss): #corresponding to ordinal classification methods        
        ordinal_goals = convert_targets_to_ordinal(bucket_goals, device=device, num_buckets=num_buckets)
        loss = criterion(output, ordinal_goals.float()) #bceloss expects targets to be floats

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

      optimizer.step()
      if scheduler is not None:
        scheduler.step()

      epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def convert_targets_to_ordinal(targets, device='cpu', num_buckets=5):
  """
  helper function for converting classification targets into ordinal classification targets.
  :param targets: classification targets, should have the form batch x num goal
  :return: ordinal targets, in the form batch x num goals x num buckets -1 (since rank 0 is implicit)
  """
  batch_size, num_goals = targets.size()

  possible_targets = torch.arange(start=0, end=num_buckets).to(device)[None, None,:] #create possible targets and add dims
  possible_targets.expand(batch_size, num_goals, -1).to(device) #expand batch and goal dimensions ,
  ordinal_targets = torch.where(possible_targets <= targets[:,:,None], 1, 0) #create expanded label set via comparison
  ordinal_targets = ordinal_targets[:,:,1:] #remove the final bucket, corresponding to the final label
  return ordinal_targets

def convert_ordinal_pred_to_pred(ordinal_pred):
  """
  helper function for converting ordinal prediction targets into single prediction for each goal (ordinal argmax)
  :param pred: ordinal prediction values, should have the form batch x num goals x num_buckets
  :return: pred, in the form batch x num goals
  """
  above_threshold = torch.where(ordinal_pred >= 0.5, 1, 0)  # find locations where binary pred is above 0.5 threshold
  consecutive_above_threshold = above_threshold.cumprod(dim=2)  # clear any locations after first 0
  pred = consecutive_above_threshold.sum(dim=2)  # sum consecutive binary predictions to get result
  return pred

def calculate_mse_weight(epoch, num_epochs, initial_weight, final_weight, annealing, logistic_slope=16):
  """
  Calculates the current weighting of mse vs ce loss
  :param epoch: current epoch number
  :param num_epochs: total number of epochs
  :param initial_weight: mse_weight during first epoch
  :param final_weight: mse_weight during final epoch
  :param annealing:
  :return: current mse_weighting
  """
  if annealing is None:
    return initial_weight
  elif annealing == 'linear': #linearly increases mse_weighting from initial to final
    return initial_weight + (final_weight - initial_weight)*(epoch/(num_epochs - 1))
  elif annealing == 'logistic': #logistically increases mse_weighting from intitial to final
    return (final_weight - initial_weight)/(1 + math.exp(-1*logistic_slope*((epoch/(num_epochs - 1) - 0.5)))) + initial_weight
  else:
    return initial_weight


