"""
File for storing constants that are used elsewhere in NL to goal codebase
"""

COUNTRY_MAP = {"<pad>": 0, "<sos>": 1, "<eos>": 2,  "Yellow_A": 3, "Yellow_B": 4, "Yellow_C": 5, "Yellow_D": 6, "Blue_A": 7, "Blue_B": 8, "Blue_C": 9, "Blue_D": 10, "Green_A": 11, "Green_B": 12, "Green_C": 13, "Green_D": 14, "Green_E": 15, "Purple_A": 16, "Purple_B": 17, "Purple_C": 18, "Purple_D": 19, "Purple_E": 20, "Red_A": 21, "Red_B": 22, "Red_C": 23}
CONSTRAINT_TYPES = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "I must have troops on ": 3, "I must not have troops on ": 4,
                    "I must be able to access ": 5, "I need to protect the borders of ": 6,
                    "I need a total of at least ": 7, "I must have at least ": 8, "I must have troops on at least ": 9,
                    "I must place at least ": 10, "I must have troops on at most ": 11}

VALUE_TYPES = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'Blue': 3, 'Green': 4, 'Yellow': 5, 'Red': 6, 'Purple': 7, '1': 8,
               '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '10': 17, '11': 18, '12': 19,
               '13': 20, '14': 21}

METRIC_NAMES = ['loss', 'cumulative_ac', 'per_goal_ac', 'per_example_ac', 'per_goal_dist', 'per_bucket_accuracy', 'mse', 'weighted_acc', 'per_goal_weighted_acc']

DATASET_BASE_PATH = '/Output_Data' #only works on linux

DATASET_CODE_TO_NAME = {'s':'synthetic', 'sa':'syn_aug', 'h':'human', 'ha':'human_augmented'} #converts training_codes to their corresponding dataset names

BERT_TO_HIDDEN_DIM = {True:768, False: 0}

SELECTION_TYPE_TO_HIDDEN_DIM = {None: 0, 'full_one_hot_no_map':168, 'full_one_hot_with_map' : 504,
                                'partial_one_hot_with_troops_no_map':21, 'partial_one_hot_with_troops_with_map':63,
                                'text_selections_no_map':0, 'text_selections_simple_map':0,
                                'text_selections_map':0}

NON_UNIFORM_BUCKETS = {5: [[-100, -52, -2, 25, 64, 100],
                           [-100, -77, -26, 24, 57, 100],
                           [-100, -3, 52, 76, 98, 100],
                           [-100, -38, 1, 40, 77, 100],
                           [-100, 22, 53, 75, 97, 100],
                           [-100, -23, 2, 47, 73, 100]],
                       3: [[-100, -21, 45, 100],
                           [-100, -42, 30, 100],
                           [-100, 41, 87, 100],
                           [-100, -1, 50, 100],
                           [-100, 54, 97, 100],
                           [-100, -1, 43, 100]]}