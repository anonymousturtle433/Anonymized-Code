from NL_to_goals.models.test_bert_to_goal import BertToGoalClass
from NL_to_goals.models.test_bert_to_goal import Classifier as BertClassifier
from NL_to_goals.Utils.training_utils import *
from NL_to_goals.Utils.evaluation_utils import *

def get_config_from_name(model_name):
  """
  Takes in a goal model name and creates a config for it that can be used to create a new model with the same parameters
  Assumes that the model is saved in the current directory
  :param model_name: string of model name
  :return: configuration dictionary
  """
  config_dict = {val.split('-')[0]: val.split('-')[1] for val in model_name.split(',')}
  config_dict['saved_epoch'] = config_dict['saved_epoch'][:-3]
  for key, val in config_dict.items():
    if key == 'lr':
      config_dict[key] = float(config_dict[key])
    else:
      try:
        config_dict[key] = int(config_dict[key])
      except:
        pass
  return config_dict

def load_bert_model(model_name, model_config):
  """
  Create a new model from configuration and loads weights from model_name into the model
  Assumes that the model is saved in the current directory
  :param model_path:
  :param model_name
  :param model_config:
  :return:
  """
  bert_model = AutoModel.from_pretrained("roberta-base")
  # set correct tokenizer
  current_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

  if model_config['g_tok'] == 'True':  # add tokens for each goal if using that approach
    current_tokenizer.add_tokens(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])
    goal_tokens = True
  else:
    goal_tokens = False

  if model_config['crit'] == 'ord':
    is_ordinal = True
  else:
    is_ordinal = False

  if model_config['coral'] == 'True':
    is_coral = True
  else:
    is_coral = False

  classifier = BertClassifier(model_config['hidden_state_size'],
                              model_config['num_goals'],
                              model_config['buckets'],
                              model_config['dropout'],
                              is_ordinal=is_ordinal,
                              is_coral=is_coral)

  selection_encoder = None  # assumes that text encoding is being used
  # initialize model weights just for untrained head
  print('Creating model')
  model = BertToGoalClass(bert_model, classifier, selection_encoder, goal_tokens=goal_tokens)

  bert_model.resize_token_embeddings(len(current_tokenizer))  # make sure that bert embedding is expanded if using goal tokens

  print('Loading model')
  model_dict = torch.load(model_name, map_location=torch.device('cpu'))
  model.load_state_dict(model_dict)

  return model, current_tokenizer

def generate_human_test_outputs(dataset,
                                eval_lists,
                                range_limit=100,
                                num_buckets=5,
                                device='cpu',
                                uniform_buckets=True):
  """
  Given gaols model (classification) and human test data, generate model responses for test dataset
  :return: dict- map number: number of goals correctly predicted, dictionary- map number: predictions
  """
  outputs, predictions, eval_list_targets = eval_lists

  maps = []
  targets = []
  for el in dataset:
    maps.append(el[3])
    goals = torch.tensor(el[2]).to(device).float()
    bucket_targets = training_utils.real_to_buckets(goals, range_limit, num_buckets, uniform_buckets=uniform_buckets,
                                                    device=device).to(device)
    targets.append(bucket_targets)

  map_to_score = {}
  map_to_predictions = {}
  for i, (target, eval_targets) in enumerate(zip(targets, eval_list_targets)):
    # check that eval list is in the same order as the maps
    assert [t.item() for t in target] == [eval_target.item() for eval_target in eval_targets], "Mismatch between dataset and eval list order"
    map_score = int(torch.sum(torch.where((eval_list_targets[i] - predictions[i]) == 0, torch.ones([1], device=device), torch.zeros([1], device=device))).item())  # gets the number of goals that were correctly bucketed for this map
    map_to_score[maps[i]] = map_score
    map_to_predictions[maps[i]] = [pred.item() for pred in predictions[i]]

  return map_to_score, map_to_predictions
