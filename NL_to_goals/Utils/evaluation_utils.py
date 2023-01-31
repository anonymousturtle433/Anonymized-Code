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
from tqdm import tqdm
from torch.nn.functional import mse_loss

from NL_to_goals.Utils import training_utils
from NL_to_goals.Utils.constants import METRIC_NAMES

from NL_to_goals.models.test_rnn_to_goal import Seq2GoalClass, Seq2GoalValue
from NL_to_goals.models.test_bert_to_goal import BertToGoal, BertToGoalValue, BertToGoalClass, GoalLabelSmoothingLoss, GoalLabelSmoothingLoss


"""
All evaluation functions (other than evaluation_manager) utilize the same function function signature to simplify the process of calling them.
The parameters are shown here instead of in the functions (to reduce repeats of this block of text)
terms of how many examples the model predicted all goals correctly
:param model: pytorch model
:param iterator: data to evaluate
:param criterion: pytorch criterion being used to optimize the model. Used as a proxy for model type (class vs regression)
:param range: upper/lower limit within which the goal values fall
:param num_buckets: number of buckets to evaluate on
:param device: device which model and data reside on
:return: percentage of examples in which the model predicting all goals accurately
"""

def evaluation_manager(model,
                       data_loaders,
                       optimizer,
                       criterion,
                       device,
                       metrics,
                       range_limit=100,
                       evaluation_bucket_nums=[5,3],
                       uniform_buckets=True,
                       mse_weight=0,
                       decay_rate=0.3,
                       dist_threshold=2):
  """
  handles the calling of evaluation metrics
  :param model: pytorch model being evaluated
  :param data_loaders: tuple containing training data_loader and validation data_loader)
  :param optimizer: pytorch optimizer being used to optimize the model
  :param criterion: pytorch loss function being used to update the model
  :param device: device that the model and the data are on
  :param metrics: list of evaluation metrics to call. Typically all evaluation metrics are called
  :param range_limit: The maximum (positive or negative) integer value that the goal data can take on.
  :param evaluation_bucket_nums: The number of classification buckets to evaluate accuracy on. Defaults to 5 and 3
  :return: two dictionaries, one for training and one for validation. Keys are the metric names and the values are a
  list of quantities corresponding to the evaluation bucket nums passed as arguments to the function
  """
  train_dataloader = data_loaders[0]
  val_dataloader = data_loaders[1]

  validation_metric_results = {metric:[] for metric in metrics}
  training_metric_results = {metric: [] for metric in metrics}

  for num_buckets in evaluation_bucket_nums:
    if num_buckets != 0:
      # generate tensors for model outputs/predictions and corresponding targets
      train_eval_lists = generate_eval_lists(model,
                                             train_dataloader,
                                             range_limit,
                                             num_buckets,
                                             device,
                                             uniform_buckets=uniform_buckets)

      valid_eval_lists = generate_eval_lists(model,
                                             val_dataloader,
                                             range_limit,
                                             num_buckets,
                                             device,
                                             uniform_buckets=uniform_buckets)

      for metric in metrics:
        if metric == 'loss':
          eval_function = evaluate
        elif metric == 'cumulative_ac':
          eval_function = evaluate_cumulative_accuracy
        elif metric == 'per_goal_ac':
          eval_function = evaluate_accuracy_per_goal
        elif metric == 'per_example_ac':
          eval_function = evaluate_accuracy_per_example
        elif metric == 'per_goal_dist':
          eval_function = evaluate_distance_per_goal
        elif metric == 'per_bucket_accuracy':
          eval_function = evaluate_accuracy_per_bucket
        elif metric == 'mse':
          eval_function = evaluate_mse
        elif metric == 'weighted_acc':
          eval_function = evaluate_weighted_accuracy_score
        elif metric == 'per_goal_weighted_acc':
          eval_function = evaluate_per_goal_weighted_accuracy_score

        eval_args = {'model':model,
                     'eval_lists':train_eval_lists,
                     'optimizer':optimizer,
                     'criterion':criterion,
                     'range':range_limit,
                     'num_buckets':num_buckets,
                     'device':device,
                     'uniform_buckets':uniform_buckets,
                     'mse_weight':mse_weight,
                     'decay_rate':decay_rate,
                     'dist_threshold':dist_threshold}

        train_quantity = eval_function(**eval_args)

        eval_args['eval_lists'] = valid_eval_lists #replace training info with validation information
        val_quantity = eval_function(**eval_args)

        training_metric_results[metric].append(train_quantity)
        validation_metric_results[metric].append(val_quantity)

  return training_metric_results, validation_metric_results


def record_metrics(writer, metrics, evaluation_bucket_nums, training_metric_results, validation_metric_results, step=0, num_goals=6):
  """
  Helper function for writing all recorded results to tensorboard
  :param writer: tensorboard writer
  :param metrics: list of names of the metrics that have been calculated
  :param evaluation_bucket_nums: list of number of buckets that the model has been evaluated on
  :param step: x axis value for the collected data
  :param training_metric_results: dictionary of results for training data generated by evaluation manager
  :param validation_metric_results: dictionary of results for validation data generated by evaluation manager
  """
  for metric in metrics:
    for i, num_buckets in enumerate(evaluation_bucket_nums):
      if num_buckets != 0:
        if metric == 'loss' or metric == 'mse':
          writer.add_scalar(f'{metric.capitalize()}/Train_{num_buckets}', training_metric_results[metric][i], step)
          writer.add_scalar(f'{metric.capitalize()}/Val_{num_buckets}', validation_metric_results[metric][i], step)
        elif metric == 'per_goal_ac' or metric == 'per_goal_dist' or metric == 'per_goal_weighted_acc':
          for goal in range(num_goals):
            writer.add_scalar(metric.capitalize() + '/Train' + f'/{num_buckets}_Bins' + '/Goal_' + str(goal + 1), training_metric_results[metric][0][goal].cpu().item(), step)
            writer.add_scalar(metric.capitalize() + '/Val' + f'/{num_buckets}_Bins' + '/Goal_' + str(goal + 1), validation_metric_results[metric][0][goal].cpu().item(), step)
        elif metric == 'per_bucket_accuracy':
          for bucket in range(num_buckets):
            writer.add_scalar(metric.capitalize() + '/Train' + f'/{num_buckets}_Bins' + '/Bucket_' + str(bucket + 1), training_metric_results[metric][0][bucket].cpu().item(), step)
            writer.add_scalar(metric.capitalize() + '/Val' + f'/{num_buckets}_Bins' + '/Bucket_' + str(bucket + 1), validation_metric_results[metric][0][bucket].cpu().item(), step)
        else:  # currently per example and cumulative accuracies only
          writer.add_scalar(metric.capitalize() + '/Train' + f'/{num_buckets}_Bins', training_metric_results[metric][i],step)
          writer.add_scalar(metric.capitalize() + '/Val' + f'/{num_buckets}_Bins', validation_metric_results[metric][i],step)

def print_metrics(epoch, training_metric_results, validation_metric_results, epoch_mins=None, epoch_secs=None):
  """
  prints the current model performance to the console
  :param epoch: current epoch
  :param epoch_mins: time epoch has taken in minutes
  :param epoch_secs: additional time epoch has taken in seconds
  :param training_metric_results: dictionary of results for training data generated by evaluation manager
  :param validation_metric_results: dictionary of results for validation data generated by evaluation manager
  """
  if epoch_mins is not None and epoch_secs is not None:
    print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
  else:
    print(f'Epoch: {epoch:02}')
  print(f'\tTrain Loss: {training_metric_results["loss"][0]:.3f}')
  print(f'\tValid Loss: {validation_metric_results["loss"][0]:.3f}')
  print(f'\tTrain MSE: {training_metric_results["mse"][0]:.3f}')
  print(f'\tValid MSE: {validation_metric_results["mse"][0]:.3f}')
  print(f'\tTrain Cumulative: {training_metric_results["cumulative_ac"][0]:.3f} | '
        f'Train Per Example: {training_metric_results["per_example_ac"][0]:.3f} | '
        f'Train Per Weighted Cumulative: {training_metric_results["weighted_acc"][0]:.3f}')
  print(f'\tValid Cumulative: {validation_metric_results["cumulative_ac"][0]:.3f} | '
        f'Valid Per Example: {validation_metric_results["per_example_ac"][0]:.3f} | '
        f'Valid Per Weighted Cumulative: {validation_metric_results["weighted_acc"][0]:.3f}')
  print(f'\tTrain Per Goal: {np.array(training_metric_results["per_goal_ac"][0].cpu())}')
  print(f'\tValid Per Goal: {np.array(validation_metric_results["per_goal_ac"][0].cpu())}')
  print(f'\tTrain Per Bucket: {np.array(training_metric_results["per_bucket_accuracy"][0].cpu())}')
  print(f'\tValid Per Bucket: {np.array(validation_metric_results["per_bucket_accuracy"][0].cpu())}')
  print(f'\tTrain Per Goal Weighted: {np.array(training_metric_results["per_goal_weighted_acc"][0].cpu())}')
  print(f'\tValid Per Goal Weighted: {np.array(validation_metric_results["per_goal_weighted_acc"][0].cpu())}')

def generate_eval_lists(model: nn.Module,
                        iterator,
                        range: int,
                        num_buckets: int,
                        device,
                        uniform_buckets=True):
  """
  Creates lists of model outputs, predictions from those outputs, and actual targets from a model, an iterator, and other parameters
  :return: outputs (direct model outputs), predictions (model predictions based on model type), targets (based on parameters)
  """
  model.eval()
  outputs = []
  predictions = []
  targets = []
  with torch.no_grad():
    for _, batch in enumerate(iterator):
      text = batch[0].to(device)
      selections = batch[4].to(device)
      goals = batch[2].to(device).float()
      bucket_goals = training_utils.real_to_buckets(goals, range, num_buckets, uniform_buckets=uniform_buckets, device=device).to(device)
      output = model(text,selections)

      targets.append(bucket_goals)
      outputs.append(output)

      if (isinstance(model, BertToGoal) or isinstance(model, BertToGoalClass)):
        if model.classifier.is_ordinal:
          batch_predictions = training_utils.convert_ordinal_pred_to_pred(output).long()
        else:
          batch_predictions = torch.argmax(output, dim=2).long()

      if isinstance(model, Seq2GoalClass) :
        batch_predictions = torch.argmax(output, dim=2).long()

      if(isinstance(model, Seq2GoalValue) or isinstance(model, BertToGoalValue)):  # note that even the regression models are evaluated using the bucketing strategy
        batch_predictions = training_utils.real_to_buckets(output, range, num_buckets, uniform_buckets=uniform_buckets, device=device)
      predictions.append(batch_predictions)

  outputs = torch.cat(outputs, dim=0)
  predictions = torch.cat(predictions, dim=0)
  targets = torch.cat(targets, dim=0)

  return outputs, predictions, targets



def evaluate(model: nn.Module,
             eval_lists,
             criterion,
             num_buckets: int,
             device,
             mse_weight=1,
             **kwargs):
  """
  Given gaols model (either classification or regression) and data, evaluate the model's performance on the data in
  terms of loss
  :return: loss value for the model on the data (1 element tensor)
  """
  model.eval()

  outputs, predictions, targets = eval_lists

  if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, GoalLabelSmoothingLoss):
    loss = criterion(outputs.permute([0, 2, 1]), targets.long())  # crossentropy expects the class dimension to be in position 1
  elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):  # only works with a regression network, otherwise it makes no sense
    loss = criterion(predictions.float(), targets.float())
  elif type(criterion) == tuple:  # criterion in the form (CrossEntropy, Distance loss), only works with classification network
    cross_entropy_loss = criterion[0](outputs.permute([0, 2, 1]), targets.long())
    distance_loss = criterion[1](predictions.float(), targets.float())
    ce_weight = 1 - mse_weight
    loss = ce_weight * cross_entropy_loss + mse_weight * distance_loss
  elif isinstance(criterion, nn.BCELoss): #corresponding to ordinal classification methods
    ordinal_targets = training_utils.convert_targets_to_ordinal(targets, device=device, num_buckets=num_buckets)
    loss = criterion(outputs, ordinal_targets.float()) #bceloss expects targets to be floats

  epoch_loss = loss.item()

  return epoch_loss


def evaluate_cumulative_accuracy(model: nn.Module,
                                 eval_lists,
                                 device,
                                 **kwargs):
  """
  Given gaols model (either classification or regression) and data, evaluate the model's performance on the data in
  terms of average accuracy over all goal types and all examples
  :return: average accuracy over all goals and all examples (1 element tensor)
  """

  model.eval()
  outputs, predictions, targets = eval_lists

  evaluation = torch.sum(torch.where((targets - predictions) == 0, torch.ones([1], device=device), torch.zeros([1], device=device)))  # gets the number of goals that were correctly bucketed
  correct_goal_predictions = evaluation.item()
  total_goal_predictions = targets.numel()

  return correct_goal_predictions / total_goal_predictions


def evaluate_accuracy_per_goal(model: nn.Module,
                               eval_lists,
                               device,
                               num_goals=6,
                               **kwargs):
  """
  Given gaols model (either classification or regression) and data, evaluate the model's performance on the data in
  terms of average accuracy over all examples but seperated out by goal
  :return: tensor with average accuracy for each goals over all examples (6 element tensor)
  """
  model.eval()
  outputs, predictions, targets = eval_lists

  evaluation = torch.sum(torch.where((targets - predictions) == 0, torch.ones([1], device=device), torch.zeros([1], device=device)),axis=0)  # gets the number of goals that were correctly buckete
  correct_goal_predictions = evaluation  # should end up being 1 by num goals
  total_goal_predictions = targets.numel() / num_goals

  return correct_goal_predictions / total_goal_predictions



def evaluate_accuracy_per_example(model: nn.Module,
                                  eval_lists,
                                  device,
                                  num_goals=6,
                                  **kwargs):
  """
  Given gaols model (either classification or regression) and data, evaluate the model's performance on the data in
  terms of how many examples the model predicted all goals correctly
  :return: percentage of examples in which the model predicting all goals accurately (1 element tensor)
  """
  model.eval()
  outputs, predictions, targets = eval_lists

  correct_predictions = torch.where(targets - predictions == 0, torch.ones([1], device=device), torch.zeros([1], device=device))
  evaluation = torch.sum(torch.floor(torch.sum(correct_predictions, axis=1) / num_goals), axis=0)
  correct_example_predictions = evaluation  # between 0 and num_examples in batch
  total_example_predictions = targets.numel() / num_goals

  return correct_example_predictions / total_example_predictions


def evaluate_mse(model: nn.Module,
                 eval_lists,
                 **kwargs):

  """
  Given gaols model (assumes classification or ordinal classification) and data, evaluate the model's performance on the data in
  terms of the mse distance between the predictions and the correct labels
  :return: mse distance between prediction and label averaged over all datapoints (1 element tensor)
  """

  model.eval()
  outputs, predictions, targets = eval_lists
  mse_distance = mse_loss(predictions, targets.float())

  epoch_dist = mse_distance.item()

  return epoch_dist



def evaluate_distance_per_goal(model: nn.Module,
                               eval_lists,
                               range: int,
                               num_buckets: int,
                               num_goals=6,
                               **kwargs):
    """
    Evaluates the average distance that the models predictions are from the target value for each goal in the -100 to 100 space
    :return: average distance for each goal (6 element tensor)
    """
    model.eval()
    outputs, predictions, targets = eval_lists
    evaluation = (2 * range / num_buckets) * torch.sum(torch.abs(targets - predictions),
                                                       axis=0)  # gets the number of goals that were correctly buckete
    per_goal_distance = evaluation  # should end up being 1 by num goals
    total_goal_predictions = targets.numel() / num_goals

    return per_goal_distance / total_goal_predictions

def evaluate_accuracy_per_bucket(model: nn.Module,
                                 eval_lists,
                                 num_buckets: int,
                                 device,
                                 **kwargs):
  """
  Given gaols model (either classification or regression) and data, evaluate the model's performance on the data in
  terms of average accuracy over all examples but seperated out by bucket
  :return: tensor with average accuracy for each goals over all examples (6 element tensor)
  """
  model.eval()
  outputs, predictions, targets = eval_lists

  one_hot_bucket_goals = torch.zeros([targets.size()[0], targets.size()[1], num_buckets],
                                     device=device).long()
  one_hot_bucket_goals.scatter_(2, targets.unsqueeze(2), 1)

  one_hot_predictions = torch.zeros([targets.size()[0], targets.size()[1], num_buckets], device=device).long()
  one_hot_predictions.scatter_(2, predictions.unsqueeze(2), 1)

  # calculate the number of correct predictions summing accross batch and goal dimensions
  evaluation = torch.sum(torch.where((one_hot_bucket_goals + one_hot_predictions) == 2, torch.ones([1], device=device),
                                     torch.zeros([1], device=device)),
                         axis=(0, 1))  # gets the number of goals that were correctly buckete

  # increment correct count for each bucked
  correct_bucket_predictions = evaluation  # should end up being 1 by num buckets

  # increment total count for each bucket
  buckets, counts = torch.unique(targets, return_counts=True)
  batch_counts = torch.zeros([num_buckets], device=device).long()
  batch_counts[buckets] = counts
  num_per_bucket_in_data = batch_counts

  return correct_bucket_predictions / num_per_bucket_in_data

def evaluate_weighted_accuracy_score(model: nn.Module,
                                     eval_lists,
                                     decay_rate=0.3,
                                     dist_threshold=2,
                                     device='cpu',
                                     **kwargs):
  """
Evaluates the weighted accuracy score for model. Predictions within the correct bucket get a score of 1, and predictions near
the correct bucket get a correspondingly lower score based on their distance.
"""
  model.eval()
  outputs, predictions, targets = eval_lists

  prediction_error_dist = torch.abs(targets - predictions).float()
  prediction_error_dist = torch.where(prediction_error_dist > dist_threshold, torch.tensor(float("inf")).to(device), prediction_error_dist)
  weighted_error = torch.pow(decay_rate, prediction_error_dist)
  weighted_acc = torch.sum(weighted_error) / targets.numel()

  return weighted_acc

def evaluate_per_goal_weighted_accuracy_score(model: nn.Module,
                                              eval_lists,
                                              num_goals=6,
                                              decay_rate=0.3,
                                              dist_threshold=2,
                                              device='cpu',
                                              **kwargs):
  """
Evaluates the weighted accuracy score for model. Predictions within the correct bucket get a score of 1, and predictions near
the correct bucket get a correspondingly lower score based on their distance.
"""
  model.eval()
  outputs, predictions, targets = eval_lists

  prediction_error_dist = torch.abs(targets - predictions).float()
  prediction_error_dist = torch.where(prediction_error_dist > dist_threshold, torch.tensor(float("inf")).to(device), prediction_error_dist)
  weighted_error = torch.pow(decay_rate, prediction_error_dist)
  per_goal_weighted_acc = torch.sum(weighted_error, dim=0)
  total_goal_predictions = targets.numel() / num_goals
  per_goal_weighted_acc = per_goal_weighted_acc / total_goal_predictions

  return per_goal_weighted_acc
