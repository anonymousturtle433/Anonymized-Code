from generate_utils import *
import argparse
import time
from tqdm import tqdm
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--num_strategies", type=int, help="The number of strategies to generate", default=10000)
parser.add_argument("--apply_augmentation", type=bool, default=False, help='Whether to apply pegasus augmentation to the data')
parser.add_argument("--aug_sentence_percent", type=float,  default=0.5,  help='Percent of sentences in input that should be augmented')
parser.add_argument("--augs_per_sentence", type=int,  default=15,  help='Number of potential augmentations to generate per sentence if augmentation turned on')
parser.add_argument("--augs_per_strategy", type=int,  default=1,  help='UNUSED Number of augmentations to generate per stratgy if augmentation turned on')
parser.add_argument("--percent_aug_data", type=float,  default=0.5,  help='Percentage of generated strategies to apply augmentation to if augmentation turned on')
parser.add_argument("--include_original", type=bool, default=True, help='If augmenting, whether to include the original data in the output')
parser.add_argument("--generate_comparison", type=bool, default=False, help='Used for generating a dataset that compares unaugmented and augmented responses')
parser.add_argument("--generics_prob", type=float,  default=0.8,  help='Initial likelihod of adding a generic sentence to strategies; decays by a factor of 2 after each setntence')
parser.add_argument("--max_generics", type=int,  default=3,  help='Maximum number of generic sentences that can be included in a strategy')
parser.add_argument("--testing", type=bool, default=False, help='Used when testing to ensure that saved data is not overridden')
parser.add_argument("--min_edit_percent", type=float,  default=0.15,  help='Minimum percentage length of original sentence that augmentations edits must meet')


args = parser.parse_args()



GC_temp = GC_Templates()
GC_temp.load_templates()

#select save path based on whether augmentation is being performed
if args.testing:
  print('Generating test csv')
  save_path = "../Data/Test_Augmented_Synthetic_Strategies.csv"
  saved_aug_info = defaultdict(list)
elif args.apply_augmentation:
    print('Generating augmented strategies')
    save_path = "../Data/Augmented_Synthetic_Strategies.csv"
    saved_aug_info = defaultdict(list)
elif args.generate_comparison:
    print('Generate Strategies for comparison')
    save_path = "../Data/Augmented_To_Synthetic_Strategy_Comparison.csv"
    saved_aug_info = defaultdict(list)
else:
    print('Generating  strategies')
    save_path = "../Data/Synthetic_Strategies.csv"
    save_aug_info = None

with open(save_path, mode='w', encoding='utf8') as output_file:
    # output_file = csv.open("../Data/Generated_Strategy.csv")
    writer = csv.writer(output_file)
    writer.writerow(
        ["Instructions", "G1_1", "G2_1", "G3_1", "G4_1", "G5_1", "G6_1", "constraint_1", "constraint_2",
         "constraint_3", "constraint_4",
         "constraint_5", "constraint_6", "constraint_7", "constraint_8"])

if args.apply_augmentation or args.generate_comparison: #instantiate pegasus model
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

num_strategies = args.num_strategies
for n in tqdm(range(num_strategies)):
    goal_assignments = []
    for g in goals.keys():
        g_assignment = random.randint(-100,100)
        goal_assignments.append((g, g_assignment))

    constraint_assignments = []
    stop = False
    count = 0
    prob_threshold = 0.8
    while not stop:
        if count < 3:
            constraint = choose_random_constraint()
            if constraint[0] > 4 and check_constraint_class(constraint, constraint_assignments):
                # check this
                continue
            if constraint not in constraint_assignments:
                constraint_assignments.append(constraint)
                count += 1
        else:
            if count == 8:
                stop = True
            elif random.uniform(0,1) < prob_threshold:
                constraint = choose_random_constraint()
                if constraint[0] > 4 and check_constraint_class(constraint, constraint_assignments):
                    continue
                if constraint not in constraint_assignments:
                    constraint_assignments.append(constraint)
                    count += 1
                prob_threshold /= 2
            else:
                stop = True

    goals_lang = GC_temp.generate_language_per_goal(goal_assignments)
    constraint_lang, constraint_assignments = GC_temp.generate_language_per_constraint(constraint_assignments)
    # print(constraint_lang)
    generics_lang = GC_temp.generate_generics(args.generics_prob, args.max_generics)

    if not args.generate_comparison:
      if random.uniform(0,1) < args.percent_aug_data and args.apply_augmentation:

          #currently appears to be saving an unaugmented strategy and then then augmented strategies
          #need to fix this before next run of creating synthetic data
          saved_aug_info = save_augmented_strategy(goal_assignments,
                                  constraint_assignments,
                                  goals_lang,
                                  constraint_lang,
                                  generics_lang,
                                  model,
                                  tokenizer,
                                  saved_aug_info,
                                  augs_per_sentence=args.augs_per_sentence,
                                  augs_per_strategy=args.augs_per_strategy,
                                  aug_sentence_percent=args.aug_sentence_percent,
                                  min_edit_percent=args.min_edit_percent,
                                  torch_device=torch_device,
                                  save_path=save_path)
      else:
          save_strategy(goal_assignments,
                        constraint_assignments,
                        goals_lang,
                        constraint_lang,
                        generics_lang,
                        save_path=save_path)
    else:
      save_strategy(goal_assignments,
                    constraint_assignments,
                    goals_lang,
                    constraint_lang,
                    generics_lang,
                    comparison_on=args.generate_comparison,
                    save_path=save_path)
      saved_aug_info = save_augmented_strategy(goal_assignments,
                              constraint_assignments,
                              goals_lang,
                              constraint_lang,
                              generics_lang,
                              model,
                              tokenizer,
                              saved_aug_info,
                              comparison_on=args.generate_comparison,
                              augs_per_sentence=args.augs_per_sentence,
                              augs_per_strategy=args.augs_per_strategy,
                              aug_sentence_percent=args.aug_sentence_percent,
                              min_edit_percent=args.min_edit_percent,
                              torch_device=torch_device,
                              save_path=save_path)


# saving for when we need to apply augmentation to human data (same workflow will be used)
# if args.apply_augmentation:
#     print('Augmenting')
#
#     # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # model_name = 'tuner007/pegasus_paraphrase'
#     # tokenizer = PegasusTokenizer.from_pretrained(model_name)
#     # model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
#
#     strategies = []
#     strategies_to_info = {}
#     with open("../Data/Synthetic_Strategies.csv", mode='r', encoding='utf8') as input_file:
#         reader = csv.reader(input_file, delimiter=',')
#         strat_counter = 0
#         for i, row in enumerate(reader): #add all strategies in dataset to list
#             if len(row) > 0 and i > 0: #don't include empty rows in the set of strategies
#                 strategies.append(row[0]) #append strategy and ignore goals/ constraints
#                 strategies_to_info[strat_counter] = row[1:]
#                 strat_counter += 1
#
#     augmentation_dict = run_augmentation_on_strategy_list(strategies,
#                                                           args.augs_per_sentence,
#                                                           args.augs_per_strategy,
#                                                           args.aug_sentence_percent,
#                                                           model,
#                                                           tokenizer,
#                                                           torch_device)
#
#     all_strategies_with_info = []
#     for i, strat in enumerate(strategies):
#         if args.include_original:
#             all_strategies_with_info.append([strat, *strategies_to_info[i]])
#         for aug_strat in augmentation_dict[i]:
#             all_strategies_with_info.append([aug_strat, *strategies_to_info[i]])
#
#     with open("../Data/Augmented_Synthetic_Strategies.csv", mode='w', encoding='utf8') as output_file:
#         writer = csv.writer(output_file)
#         writer.writerow(
#             ["Instructions", "G1_1", "G2_1", "G3_1", "G4_1", "G5_1", "G6_1", "constraint_1", "constraint_2",
#              "constraint_3", "constraint_4",
#              "constraint_5", "constraint_6", "constraint_7", "constraint_8"])
#         for strat_info in all_strategies_with_info:
#             writer.writerow(strat_info)