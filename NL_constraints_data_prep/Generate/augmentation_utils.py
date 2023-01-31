import torch
from torch import tensor
import math
from random import shuffle
import random
import json
import csv
import string
import edit_distance
from tqdm import tqdm

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

#used to interface with pegasus and generate results
#currently most hyperparameters are set to the tutorial defaults - it may be worth tuning these in the future
def get_response(input_text, model, tokenizer, num_return_sequences, num_beams, torch_device='cpu'):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

#inputs: path to json file with the examples to augment (assumes examples are stored using the standard project form)
#outputs: list with elements of the form (list(sentences), goals)
#outputs are used for the data augmentation

#TODO: add error handling
def parse_json(json_file_path):
    with open(json_file_path, 'rb') as f:
        train_data = json.load(f)
        train_data_splittext_goal = [(data['Text'].split(".")[:-1], data['Goals']) for data in train_data] #remove the last sentence, as it is '' which should NOT be augmented
    return train_data, train_data_splittext_goal

# filter relies on the assumption that each potential term (i.e. Blue) is only used once in each sentence
# if this assumption is violated, then it is possible for the model to lose in
def filter_augmentation_by_term(orig, aug):
    #figure out a better place to put these constants eventually
    NUMS = [str(i) for i in range(1,15)]
    WORD_NUMS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen']
    CONTINENT_COLORS = ['Yellow', 'Red', 'Blue', 'Purple', 'Green']
    PLAYER_COLORS = ['Grey', 'Black']
    COUNTRY_LETTERS = ['A', 'B', 'C', 'D', 'E']
    COUNTRIES = []
    ASSORTED_TERMS = []

    for color in CONTINENT_COLORS:
        for letter in COUNTRY_LETTERS:
            COUNTRIES.append(color + ' ' + letter)
            COUNTRIES.append(color + '_' + letter)

    PROTECTED_TERMS = NUMS + WORD_NUMS + CONTINENT_COLORS + PLAYER_COLORS + COUNTRIES + ASSORTED_TERMS
    for term in PROTECTED_TERMS:

        if (term.lower() in orig.lower()) and (term.lower() not in aug.lower()):
            return False
    return True

#currently producing weird results, so disabling for now
def filter_augmentation_by_len(orig, aug):
    orig_len = 1
    aug_len = len(aug.split('.')[:-1]) # may be more than onw
    if orig_len != aug_len:
        # print('Removing sentence for length')
        # print(orig)
        # print(aug)
        # print()
        return False
    return True

#remove augmentation's who's edit distance is less than the minimum amount requested
def filter_augmentation_by_edit_distance(orig, aug, min_edit_distance):

    orig_tokens = [token for token in orig.lower().strip().split(" ") if token != ' ']
    aug_tokens = [token for token in aug.lower().strip().split(" ") if token != ' ']
    sm = edit_distance.SequenceMatcher(a=orig_tokens, b=aug_tokens)
    if sm.distance() < min_edit_distance:
        return False
    return True

#remove augmentations that have repeated sequences of words, typically indicating a strange transformer output
def filter_repeated_sequences(aug, num_repeats=5):
    aug_words = aug.split(' ')
    repeat_count = 1
    previous_word = None
    for word in aug_words:
        if word == previous_word:
            repeat_count += 1
        else:
            repeat_count = 1
        previous_word = word
        if repeat_count >= num_repeats:
            return False
    return True


# function for generating the information needed to generate augmentations
# input: data in the form of a list of this form of element (strategy as a list of sentences without periods, goals), generation parameters
# output: data in the form of a list of this form of element (original sentences, list of lists of augmented sentences for each sentence, goals)

# note that pegasus may not generate any suitable augmentations for a given sentence - further processing is used to ignore this case
# # this could also be handled by generating additional augmentations, but this method gets complicated pretty quickly (what happens if the additional augmentations fail, etc)
# def generate_augmentation_info(data, model, tokenizer, device='cpu', num_generations=5, num_beams=5, augmentation_percentage=0.5):
#     strategy_augmentation_information = []
#     for i, sentences in enumerate(data):
#         sentence_augmentations = [[] for _ in sentences]
#         for j, sentence in enumerate(sentences):
#             if random.uniform(0, 1) < augmentation_percentage: #corrected
#                 augmentations = get_response(sentence, model, tokenizer, num_generations, num_beams, torch_device=device)
#                 # apply filters here to exclude bad augmentations
#                 augmentations = [aug.replace('.', '') for aug in augmentations if
#                                  ((filter_augmentation_by_term(sentence, aug)) and
#                                   filter_augmentation_by_edit_distance(sentence, aug, min_edit_distance=len(sentence//10)))]  #change hard coded 10 to be a usable hyperparameter
#                 if len(augmentations) == 0: #if all augmentations from pegasus filtered out, use original sentence
#                     augmentations = [sentences[j]]
#             else:
#                 augmentations = [sentences[j]]
#             sentence_augmentations[j] += augmentations
#
#         strategy_augmentation_information.append((sentences, sentence_augmentations))
#     return strategy_augmentation_information

#sent_to_aug = default_dict with sentences as keys and existing augs as values. Should reduce the number of calls to pegasus
def generate_augmentation_info_with_save_info(data,
                                              model,
                                              tokenizer,
                                              save_aug_info,
                                              device='cpu',
                                              num_generations=5,
                                              num_beams=5,
                                              augmentation_percentage=0.5,
                                              min_edit_percent=0.1,
                                              hide_progress=True):
    strategy_augmentation_information = []
    for i, sentences in tqdm(enumerate(data),disable=hide_progress):
        sentence_augmentations = [[] for _ in sentences]
        for j, sentence in enumerate(sentences):
            if random.uniform(0, 1) < augmentation_percentage: #corrected
                existing_augmentations = save_aug_info[sentence]
                if len(existing_augmentations) == 0: #if no sentences in aug dict
                    augmentations = get_response(sentence, model, tokenizer, num_generations, num_beams, torch_device=device)
                    augmentations = [aug.replace('.', '') for aug in augmentations if
                                     ((filter_augmentation_by_term(sentence, aug)) and
                                      filter_augmentation_by_edit_distance(sentence, aug, min_edit_distance=int(len(sentence)*min_edit_percent)) and
                                      filter_repeated_sequences(aug,num_repeats=5))] #currently hardcoding repeat length

                    if len(augmentations) == 0: #if all augmentations from pegasus filtered out, use original sentence
                        augmentations = [sentence]
                    save_aug_info[sentence] = augmentations
                else:
                    augmentations = existing_augmentations
            else:
                augmentations = [sentence]
            sentence_augmentations[j] += augmentations

        strategy_augmentation_information.append((sentences, sentence_augmentations))
    return strategy_augmentation_information, save_aug_info

#input -
#   augmentation_info: single element from the list produced by strategy_augmentation_information
#   sentence_replace_percentage: the number of sentences in the original text that should be replaced with an augmentation
#   num_new_examples: the number of augmented examples to generate
#output - list of text for newly generated augmentations, original text

#this function takes in the augmented information and a coule of hyperparameters, and replaces sentences in the original
#example to generate a new version of the same strategy. The randomness in the new example comes both the sentences
#that are chosen for replacement and the particular augmentations that are chosen to replace each sentence. This does
#mean that there is a chance that duplicate augmentations will be generated, but given a reasonable number of augmentations
#and sentences, this chance is extremely small
def augment_strategy(augmentation_info, sentence_replace_percentage=0.5, num_new_examples=1):
    sentences = augmentation_info[0]
    sentence_augs = augmentation_info[1]

    num_sentences = len(sentences)
    num_replaced_sentences = math.floor(num_sentences * sentence_replace_percentage)

    new_examples = []
    for _ in range(num_new_examples):
        # randomly generate the list of sentences from the original text that should be replaced with augmentations
        sentence_locations = list(range(num_sentences))
        shuffle(sentence_locations)
        replacement_locs = sentence_locations[:num_replaced_sentences]
        new_example = []
        for loc in range(num_sentences):
            if loc in replacement_locs:  # add a random augmentation in place of the original sentence
                loc_augs = sentence_augs[loc]
                if len(loc_augs) > 0:  # pegasus generated at least 1 valid augmentation
                    shuffle(loc_augs)
                    new_example.append(loc_augs[0])
            else:  # add the original sentence
                new_example.append(sentences[loc])
        new_examples.append('. '.join(new_example) + '.')
    original_text = '. '.join(sentences) + '.'
    return new_examples, original_text

def run_augmentation_on_strategy_list(strategies,
                                      augs_per_sentence,
                                      augs_per_strategy,
                                      aug_sentence_percent,
                                      min_edit_percent,
                                      model,
                                      tokenizer,
                                      torch_device,
                                      save_info,
                                      hide_progress=True):

    #changing the list of strategies into a form that is compatible with other functions
    # note that the 'None' would normally be goals, but since those aren't currently used in augmentation, ignoring them for now
    parsed_strategies = [strategy.split(".")[:-1] for strategy in strategies]

    # augmentation_info = generate_augmentation_info(parsed_strategies,
    #                                                model,
    #                                                tokenizer,
    #                                                num_generations=augs_per_sentence,
    #                                                num_beams=augs_per_sentence,
    #                                                device=torch_device,
    #                                                augmentation_percentage = aug_sentence_percent)

    augmentation_info, save_info = generate_augmentation_info_with_save_info(parsed_strategies,
                                                                               model,
                                                                               tokenizer,
                                                                               save_info,
                                                                               num_generations=augs_per_sentence,
                                                                               num_beams=augs_per_sentence,
                                                                               device=torch_device,
                                                                               augmentation_percentage=aug_sentence_percent,
                                                                               min_edit_percent=min_edit_percent,
                                                                               hide_progress=hide_progress)



    augmented_strategy_dict = {}
    for i, individual_augment_info in enumerate(augmentation_info):
        new_text, text = augment_strategy(individual_augment_info, sentence_replace_percentage=aug_sentence_percent, num_new_examples=augs_per_strategy)
        augmented_strategy_dict[i] = new_text
    return augmented_strategy_dict, save_info


# if __name__ == "__main__":
#     print('Loading models')
#     torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # torch_device = 'cpu'
#
#     model_name = 'tuner007/pegasus_paraphrase'
#     tokenizer = PegasusTokenizer.from_pretrained(model_name)
#     model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
#
#     training_data_path = "../NL_constraints_data_prep/Output_Data/nl_goals_constraints_train.json"
#
#     print('Parsing json')
#     data, parsed_data = parse_json(training_data_path)
#     # parsed_data = parsed_data[0:2] #if testing on smaller set of info
#     print('Generating augmentation data')
#     augmentation_info = generate_augmentation_info(parsed_data, model, tokenizer, num_generations=5, num_beams=5)
#
#     print('Generating augmentation Examples')
#     #naive augmentation - just generate an extra example for each original piece of data
#     comparison_list = []
#     with open('naive_augmentation.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         for i, individual_augment_info in enumerate(augmentation_info):
#             new_text, text = augment_strategy(individual_augment_info, num_new_examples=25)
#             row = [text] + new_text + individual_augment_info[2]
#             writer.writerow(row)











