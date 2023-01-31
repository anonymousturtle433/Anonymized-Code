from NL_constraints_data_prep.Utils.Dataset import setup_datasets, prepare_data
from NL_to_goals.models.test_bert_to_goal import BertToGoalClass
from NL_to_goals.models.test_bert_to_goal import Classifier as BertClassifier
from NL_to_goals.Utils.training_utils import *
from NL_to_goals.Utils.evaluation_utils import *
from NL_to_goals.Utils.constants import *
from NL_to_goals.evaluate_goal_model.evaluate_goal_model_helpers import *
from functools import partial
from tqdm import tqdm
import json
import csv
import os

#getting model name. Change the model name will all the parameters it should have. Check the evaluate_goal_model_helpers
model_name = None
for root, dirs, files in os.walk(".", topdown=False):
    full_root = root
    for name in files:
        if '.pt' in name:
            model_name = name
            break

assert model_name is not None, 'No model found'
print(f"Model Found: {model_name}")

#set up config for model
config = get_config_from_name(model_name)
config['num_goals'] = 6
config['hidden_state_size'] = 768
config['dropout'] = 0.5

#load model, get tokenizer
loaded_model, tokenizer = load_bert_model(model_name, config)

#set up datasets and dataloaders
datasets = 'human'
BATCH_SIZE = 8
selection_type = config['sel']
if config['g_tok'] == 'True':  # add tokens for each goal if using that approach
    goal_tokens = True
else:
    goal_tokens = False

train_dataset, valid_dataset, test_dataset = setup_datasets(datasets, include_unk=True, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, tokenizer=tokenizer, selection_type=selection_type, goal_tokens=goal_tokens), shuffle=True)
val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, tokenizer=tokenizer, selection_type=selection_type, goal_tokens=goal_tokens))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, tokenizer=tokenizer, selection_type=selection_type, goal_tokens=goal_tokens))

#generate eval lists for use in evaluating cumulative accuracy
print('Generating validation eval lists')
valid_eval_lists = generate_eval_lists(loaded_model,
                                     val_dataloader,
                                     100,
                                     config['buckets'],
                                     'cpu',
                                     uniform_buckets=True)

print('Generating test eval lists')
test_eval_lists = generate_eval_lists(loaded_model,
                                     test_dataloader,
                                     100,
                                     config['buckets'],
                                     'cpu',
                                     uniform_buckets=True)
#calculates accuracies
valid_acc = evaluate_cumulative_accuracy(loaded_model, valid_eval_lists, 'cpu')
test_acc = evaluate_cumulative_accuracy(loaded_model, test_eval_lists, 'cpu')
print('Please check that validation accuracy matches expectations to confirm that you are testing the correct checkpoint')
print(f"Validation accuracy: {valid_acc}")
print(f"Test set accuracy: {test_acc}")

#generate map to score and map to prediction information
map_to_score, map_to_predictions = generate_human_test_outputs(test_dataset, test_eval_lists)

#save the val and test accs
with open(f"eval_results_{model_name}.json", "w") as f:
  json.dump({'val_acc':valid_acc, 'test_acc':test_acc}, f)

#save map to score
with open(f"map_to_score_{model_name}.json", "w") as f:
  json.dump(map_to_score, f)

#save map to prediction
with open(f"map_to_predictions_{model_name}.csv", mode='w', encoding='utf8') as f:
  writer = csv.writer(f)
  writer.writerow(["Map", "G1_1", "G2_1", "G3_1", "G4_1", "G5_1", "G6_1"])
  for current_map, predictions in map_to_predictions.items():
    writer.writerow([current_map] + predictions)
