from generate_utils import *
import argparse
import json
from NL_constraints_data_prep.Generate.augmentation_utils import run_augmentation_on_strategy_list, augment_strategy

parser = argparse.ArgumentParser()
parser.add_argument("--aug_sentence_percent", type=float,  default=1,  help='Percent of sentences in input that should be augmented')
parser.add_argument("--augs_per_sentence", type=int,  default=15,  help='Number of potential augmentations to generate per sentence if augmentation turned on')
parser.add_argument("--augs_per_strategy", type=int,  default=1,  help='Number of augmentations to generate per stratgy if augmentation turned on')
parser.add_argument("--percent_aug_data", type=float,  default=1,  help='Percentage of generated strategies to apply augmentation to if augmentation turned on')
parser.add_argument("--include_original", type=bool, default=True, help='If augmenting, whether to include the original data in the output')
parser.add_argument("--min_edit_percent", type=float,  default=0.15,  help='Minimum percentage length of original sentence that augmentations edits must meet')


args = parser.parse_args()

print("Loading pegasus model")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

print('Augmenting human training data')
#apply augmentation to human training data
#assumes that the human json data is already in the Output_Data folder
strategy_texts = []
strategies_to_info = {}


with open("../Output_Data/nl_goals_constraints_val_human.json", mode='r', encoding='utf8') as input_file:
  validation_data = json.load(input_file)
with open("../Output_Data/nl_goals_constraints_test_human.json", mode='r', encoding='utf8') as input_file:
  test_data = json.load(input_file)

with open("../Output_Data/nl_goals_constraints_train_human.json", mode='r', encoding='utf8') as input_file:
  original_strategies = json.load(input_file)
  for i, strat in enumerate(original_strategies): #add all strategies in dataset to list
    strategy_texts.append(strat['Text']) #append strategy and ignore goals/ constraints

print('Augmenting Strategies')
augmentation_dict, _ = run_augmentation_on_strategy_list(strategy_texts,
                                                          args.augs_per_sentence,
                                                          args.augs_per_strategy,
                                                          args.aug_sentence_percent,
                                                          args.min_edit_percent,
                                                          model,
                                                          tokenizer,
                                                          torch_device,
                                                          defaultdict(list),
                                                          hide_progress=False)

augmented_training_data = []
for i, strat in enumerate(original_strategies):
  if args.include_original:
      augmented_training_data.append(original_strategies[i])
  for aug_strat in augmentation_dict[i]:
      updated_strat = strat.copy()
      updated_strat['Text'] = aug_strat
      augmented_training_data.append(updated_strat)

with open("../Output_Data/nl_goals_constraints_train_human_aug.json", 'w', encoding='utf8') as output_file:
    json.dump(augmented_training_data, output_file)
with open("../Output_Data/nl_goals_constraints_val_human_aug.json", mode='w', encoding='utf8') as output_file:
  json.dump(validation_data, output_file)
with open("../Output_Data/nl_goals_constraints_test_human_aug.json", mode='w', encoding='utf8') as output_file:
  json.dump(test_data, output_file)

#save json