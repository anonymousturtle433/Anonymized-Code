import os
import math
import random
from .data_utils_helpers import *
from torch.utils.data import DataLoader
from torchtext.vocab import vocab as torchtext_vocab
from torchtext.vocab import build_vocab_from_iterator as tt_build_vocab_from_iterator
import json
import pandas as pd
import argparse
import json
from NL_constraints_data_prep.Generate.augmentation_utils import run_augmentation_on_strategy_list, augment_strategy


logger = logging.getLogger(__name__)



class NLConstraintsDataset(torch.utils.data.Dataset):
    """Defines a data set for language, goals and constraints.
	"""

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.
		Arguments:
		vocab: Vocabulary object used for dataset.
		data: a list of label/tokens tuple. tokens are a tensor after
		    numericalizing the string tokens. label is an integer.
		    [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
		label: a set of the labels.
		{label1, label2}
		Examples:
			See the examples in examples/text_classification/
		"""

        super(NLConstraintsDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        return self._vocab

    def get_labels(self):
        return self._labels

def prepare_data(datasets, kfold = False, fold = None):
    if 'human_augmented' in datasets:
        augment = True
    else:
        augment = False
    if not kfold:
        if not augment:
            load_data(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Data', datasets)
        data_prep(datasets, augment = augment)
    else:
        data_prep_10fold(datasets, fold = fold, augment = augment)

def setup_datasets(datasets, output_dir='/Output_Data/', vocab=None, include_unk=False, tokenizer = None, kfold = False):
    footer = ''
    if 'v1' in datasets or 'v2' in datasets or 'v3' in datasets or 'human' in datasets:
        footer += '_human'
    if 'syn' in datasets:
        footer += '_syn'
    if 'aug' in datasets:
        footer += '_aug'
    
    
    if 'human_augmented' in datasets:
        # This additional if branch is necessary for the human_augmented condition due to the difference in the datasets input for the goals and constraints model
        # For the goals model, 'datasets' is a string corresponding to the specific dataset, e.g. 'human_augmented'
        # For the constraints model, 'datasets' is a list of strings corresponding to all the datasets to be included, e.g. ['human_augmented']
        footer += '_human_aug'
     
    

    if kfold:
        footer += '_kfold'

    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + output_dir
    train_json_path = output_dir + f'nl_goals_constraints_train{footer}.json'
    test_json_path = output_dir + f'nl_goals_constraints_test{footer}.json'
    val_json_path = output_dir + f'nl_goals_constraints_val{footer}.json'

    if vocab is None and tokenizer == None:
        logging.info('Building Vocab based on {}'.format(train_json_path))
        vocab = tt_build_vocab_from_iterator(json_iterator(train_json_path), min_freq=4, specials=['<pad>', '<unk>', '<sos>', '<eos>'])
    elif vocab is None and tokenizer != None:
        logging.info('Building Vocab based on passed tokenizer')
        vocab = torchtext_vocab(Counter(tokenizer.get_vocab()))
    else:
        if not isinstance(vocab, torchtext_vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data - ',train_json_path)
    vocab.set_default_index(vocab['<unk>'])
    train_data, train_labels = create_data_from_iterator(
        vocab, json_iterator(train_json_path, yield_cls=True, tokenizer = tokenizer), include_unk, tokenizer = tokenizer)
    print('Creating validation data - ', val_json_path)
    val_data, val_labels = create_data_from_iterator(
        vocab, json_iterator(val_json_path, yield_cls=True, tokenizer = tokenizer), include_unk, tokenizer = tokenizer)
    print('Creating testing data - ', test_json_path)
    test_data, test_labels = create_data_from_iterator(
        vocab, json_iterator(test_json_path, yield_cls=True, tokenizer = tokenizer), include_unk, tokenizer = tokenizer)
    # if len(train_labels ^ test_labels) > 0:
    # 	raise ValueError("Training and test labels don't match")
    return (NLConstraintsDataset(vocab, train_data, train_labels),
            NLConstraintsDataset(vocab, val_data, val_labels),
            NLConstraintsDataset(vocab, test_data, test_labels))


def load_data(output_dir, datasets = []):
    assert 'human_augmented' not in datasets, "This function should not be called for the human augmented dataset."
    def drop_prefix(self, prefix):
        self.columns = self.columns.str.lstrip(prefix)
        return self

    def drop_suffix(self, prefix):
        self.columns = self.columns.str.rstrip(prefix)
        return self

    pd.core.frame.DataFrame.drop_prefix = drop_prefix
    pd.core.frame.DataFrame.drop_suffix = drop_suffix
    # output_dir = os.path.dirname(os.path.abspath(__file__)) + data_dir
    for subdir, dirs, files in os.walk(output_dir):
        line_count = 0
        row = 1
        all_dfs = []
        print(datasets)
        for filename in files:
            if filename.endswith(".xlsx"):
                if 'v1' in filename and 'v1' in datasets:
                    df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
                    M1 = ['Q31', 'A_select', 'A_quantity', 'A_constraint', 'M1_G']
                    M2 = ['Q73', 'B_select', 'B_quantity', 'B_constraint', 'M2_G']
                    M3 = ['Q40', 'C_select', 'C_quantity', 'C_constraint', 'M3_G']
                    M4 = ['Q45', 'D_select', 'D_quantity', 'D_constraint', 'M4_G']
                    M5 = ['Q50', 'E_select', 'E_quantity', 'E_constraint', 'M5_G']

                    m1_df = df.loc[:, df.columns.str.contains('|'.join(M1))].drop(index=0)
                    m1_df = m1_df.dropna(axis=0, how='all')
                    m1_df.insert(0, "M1_map", [1] * len(m1_df))
                    m1_df = m1_df.rename(columns={'Q31': 'M1_Instructions'})
                    m2_df = df.loc[:, df.columns.str.contains('|'.join(M2))].drop(index=0)
                    m2_df = m2_df.dropna(axis=0, how='all')
                    m2_df.insert(0, "M2_map", [2] * len(m2_df))
                    m2_df = m2_df.rename(columns={'Q73': 'M2_Instructions'})
                    m3_df = df.loc[:, df.columns.str.contains('|'.join(M3))].drop(index=0)
                    m3_df = m3_df.dropna(axis=0, how='all')
                    m3_df.insert(0, "M3_map", [3] * len(m3_df))
                    m3_df = m3_df.rename(columns={'Q40': 'M3_Instructions'})
                    m4_df = df.loc[:, df.columns.str.contains('|'.join(M4))].drop(index=0)
                    m4_df = m4_df.dropna(axis=0, how='all')
                    m4_df.insert(0, "M4_map", [4] * len(m4_df))
                    m4_df = m4_df.rename(columns={'Q45': 'M4_Instructions'})
                    m5_df = df.loc[:, df.columns.str.contains('|'.join(M5))].drop(index=0)

                    m5_df = m5_df.dropna(axis=0, how='all')
                    m5_df.insert(0, "M5_map", [5] * len(m5_df))
                    m5_df = m5_df.rename(columns={'Q50': 'M5_Instructions'})
                    # m1_df.append(m2_df, ignore_index = True)
                    m1_df.drop_prefix('A_')
                    m1_df.drop_prefix('M1_')
                    m2_df.drop_prefix('B_')
                    m2_df.drop_prefix('M2_')
                    m3_df.drop_prefix('C_')
                    m3_df.drop_prefix('M3_')
                    m4_df.drop_prefix('D_')
                    m4_df.drop_prefix('M4_')
                    m5_df.drop_prefix('E_')
                    m5_df.drop_prefix('M5_')
                    all_dfs.extend([m1_df, m2_df, m3_df, m4_df, m5_df])
                elif 'v2' in filename and 'v2' in datasets:
                    df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
                    count = 0
                    # check which file you are looking at and decide which dataframes to append to.
                    M6 = ['Q31', 'F_select', 'F_quantity', 'F_constraint', 'M6_G']
                    M7 = ['Q73', 'G_select', 'G_quantity', 'G_constraint', 'M7_G']
                    M8 = ['Q40', 'H_select', 'H_quantity', 'H_constraint', 'M8_G']
                    M9 = ['Q45', 'I_select', 'I_quantity', 'I_constraint', 'M9_G']
                    M10 = ['Q50', 'J_select', 'J_quantity', 'J_constraint', 'M10_G']
                    m6_df = df.loc[:, df.columns.str.contains('|'.join(M6))].drop(index=0)
                    m6_df = m6_df.dropna(axis=0, how='all')
                    m6_df.insert(0, "M6_map", [6] * len(m6_df))
                    m6_df = m6_df.rename(columns={'Q31': 'M6_Instructions'})
                    m7_df = df.loc[:, df.columns.str.contains('|'.join(M7))].drop(index=0)
                    m7_df = m7_df.dropna(axis=0, how='all')
                    m7_df.insert(0, "M7_map", [7] * len(m7_df))
                    m7_df = m7_df.rename(columns={'Q73': 'M7_Instructions'})
                    m8_df = df.loc[:, df.columns.str.contains('|'.join(M8))].drop(index=0)
                    m8_df = m8_df.dropna(axis=0, how='all')
                    m8_df.insert(0, "M8_map", [8] * len(m8_df))
                    m8_df = m8_df.rename(columns={'Q40': 'M8_Instructions'})
                    m9_df = df.loc[:, df.columns.str.contains('|'.join(M9))].drop(index=0)
                    m9_df = m9_df.dropna(axis=0, how='all')
                    m9_df.insert(0, "M9_map", [9] * len(m9_df))
                    m9_df = m9_df.rename(columns={'Q45': 'M9_Instructions'})
                    m10_df = df.loc[:, df.columns.str.contains('|'.join(M10))].drop(index=0)

                    m10_df = m10_df.dropna(axis=0, how='all')
                    m10_df.insert(0, "M10_map", [10] * len(m10_df))
                    m10_df = m10_df.rename(columns={'Q50': 'M10_Instructions'})
                    # m1_df.append(m2_df, ignore_index = True)
                    m6_df.drop_prefix('F_')
                    m6_df.drop_prefix('M6_')
                    m7_df.drop_prefix('G_')
                    m7_df.drop_prefix('M7_')
                    m8_df.drop_prefix('H_')
                    m8_df.drop_prefix('M8_')
                    m9_df.drop_prefix('I_')
                    m9_df.drop_prefix('M9_')
                    m10_df.drop_prefix('J_')
                    m10_df.drop_prefix('M10_')
                    all_dfs.extend([m6_df, m7_df, m8_df, m9_df, m10_df])
                elif 'v3' in filename and 'v3' in datasets:
                    df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
                    count = 0
                    # check which file you are looking at and decide which dataframes to append to.
                    M11 = ['Q31', 'K_select', 'K_quantity', 'K_constraint', 'M11_G']
                    M12 = ['Q73', 'L_select', 'L_quantity', 'L_constraint', 'M12_G']
                    M13 = ['Q40', 'M_select', 'M_quantity', 'M_constraint', 'M13_G']
                    M14 = ['Q45', 'N_select', 'N_quantity', 'N_constraint', 'M14_G']
                    M15 = ['Q50', 'O_select', 'O_quantity', 'O_constraint', 'M15_G']
                    m11_df = df.loc[:, df.columns.str.contains('|'.join(M11))].drop(index=0)
                    m11_df = m11_df.dropna(axis=0, how='all')
                    m11_df.insert(0, "M11_map", [11] * len(m11_df))
                    m11_df = m11_df.rename(columns={'Q31': 'M11_Instructions'})
                    m12_df = df.loc[:, df.columns.str.contains('|'.join(M12))].drop(index=0)
                    m12_df = m12_df.dropna(axis=0, how='all')
                    m12_df.insert(0, "M12_map", [12] * len(m12_df))
                    m12_df = m12_df.rename(columns={'Q73': 'M12_Instructions'})
                    m13_df = df.loc[:, df.columns.str.contains('|'.join(M13))].drop(index=0)
                    m13_df = m13_df.dropna(axis=0, how='all')
                    m13_df.insert(0, "M13_map", [13] * len(m13_df))
                    m13_df = m13_df.rename(columns={'Q40': 'M13_Instructions'})
                    m14_df = df.loc[:, df.columns.str.contains('|'.join(M14))].drop(index=0)
                    m14_df = m14_df.dropna(axis=0, how='all')
                    m14_df.insert(0, "M14_map", [14] * len(m14_df))
                    m14_df = m14_df.rename(columns={'Q45': 'M14_Instructions'})
                    m15_df = df.loc[:, df.columns.str.contains('|'.join(M15))].drop(index=0)

                    m15_df = m15_df.dropna(axis=0, how='all')
                    m15_df.insert(0, "M15_map", [15] * len(m15_df))
                    m15_df = m15_df.rename(columns={'Q50': 'M15_Instructions'})
                    # m1_df.append(m2_df, ignore_index = True)
                    m11_df.drop_prefix('K_')
                    m11_df.drop_prefix('M11_')
                    m12_df.drop_prefix('L_')
                    m12_df.drop_prefix('M12_')
                    m13_df.drop_prefix('M_')
                    m13_df.drop_prefix('M13_')
                    m14_df.drop_prefix('N_')
                    m14_df.drop_prefix('M14_')
                    m15_df.drop_prefix('O_')
                    m15_df.drop_prefix('M15_')
                    all_dfs.extend([m11_df, m12_df, m13_df, m14_df, m15_df])
            elif filename.endswith(".csv"):
                if "Synthetic" in filename and "syn" in datasets and not 'Augmented' in filename and not "aug" in datasets: #if we only want synthetic data
                    syn_df = pd.read_csv(subdir + '/' + filename, encoding='utf8')
                    populate_synthetic_data(syn_df)
                    all_dfs.append(syn_df)
                if "Synthetic" in filename and "syn" in datasets and 'Augmented' in filename and "aug" in datasets: #if we want synthetic augmented data
                    syn_df = pd.read_csv(subdir + '/' + filename, encoding='utf8')
                    populate_synthetic_data(syn_df)
                    all_dfs.append(syn_df)
        for num, df in enumerate(all_dfs):
            if num == 0:
                final = df
            else:
                final = final.append(df, ignore_index=True)

        # move this outside the for loop after creating dataframes, append all dataframes 1 - 10 to final and create final excel file
        # final = m1_df.append(m2_df, ignore_index=True).append(m3_df, ignore_index=True).append(m4_df,
        #                                                                                        ignore_index=True).append(
        #     m5_df, ignore_index=True).append(m6_df, ignore_index=True).append(m7_df, ignore_index=True).append(m8_df,
        #                                                                                                        ignore_index=True).append(
        #     m9_df, ignore_index=True).append(m10_df, ignore_index=True)
        footer = ''
        if 'v1' in datasets or 'v2' in datasets or 'v3' in datasets or 'human' in datasets:
            footer += '_human'
        if 'syn' in datasets:
            footer += '_syn'
        if 'aug' in datasets:
            footer += '_aug'

        writer = pd.ExcelWriter(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'/Output_Data/consolidated_data{footer}.xlsx',
                                engine='xlsxwriter')
        final.to_excel(writer, sheet_name='Sheet1', index=False, encoding = 'utf8')
        writer.save()


def data_prep(datasets, output_dir='/Output_Data', augment = False):
    '''
    Create JSON file from Consolidated_data excel sheet
    Encode + pad constraints, pad selections
    Stores JSON files for train, test and valid in "output_dir"
    '''

    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + output_dir
    assert 'human_augmented' not in datasets, "This function will break if human aug is being used since we don't want to overwrite the human aug datasets."
    footer = ''
    if 'v1' in datasets or 'v2' in datasets or 'v3' in datasets or 'human' in datasets:
        footer += '_human'
    if 'syn' in datasets:
        footer += '_syn'
    if 'aug' in datasets:
        footer += '_aug'
    if 'human_augmented' in datasets:
        # This additional if branch is necessary for the human_augmented condition due to the difference in the datasets input for the goals and constraints model
        # For the goals model, 'datasets' is a string corresponding to the specific dataset, e.g. 'human_augmented'
        # For the constraints model, 'datasets' is a list of strings corresponding to all the datasets to be included, e.g. ['human_augmented']
        footer += '_human_aug'


    test_filename = f'user_study_data_points.xlsx'
    test_data = read_data(test_filename, output_dir)

    filename = f'consolidated_data{footer}.xlsx'

    data = read_data(filename, output_dir)
    train_data = data[:math.ceil(float(len(data)) * 0.85)]
    val_data = data[math.ceil(float(len(data)) * 0.85):]

    with open(output_dir + f'/nl_goals_constraints_train{footer}.json', 'w', encoding='utf8') as f:
        json.dump(train_data, f, default=convert, indent=4)
    with open(output_dir + f'/nl_goals_constraints_val{footer}.json', 'w', encoding='utf8') as f:
        json.dump(val_data, f, default=convert, indent=4)
    with open(output_dir + f'/nl_goals_constraints_test{footer}.json', 'w', encoding='utf8') as f:
        json.dump(test_data, f, default=convert, indent=4)


def data_prep_10fold(datasets, output_dir='/Output_Data', fold = 0, augment = False):
    '''
    Create json files for each fold for the 10-fold cross validation.
    Splits the entire datasets into 10 85/15 splits such that each data point is included in the validation set at least once.
    '''
    assert -1 < fold < 10, "For 10-fold cross-validation, fold value needs to be between 0-9"
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + output_dir

    footer = ''
    if 'v1' in datasets or 'v2' in datasets or 'v3' in datasets or 'human' in datasets:
        footer += '_human'
    if 'syn' in datasets:
        footer += '_syn'
    if 'aug' in datasets:
        footer += '_aug'
    if 'human_augmented' in datasets:
        # This additional if branch is necessary for the human_augmented condition due to the difference in the datasets input for the goals and constraints model
        # For the goals model, 'datasets' is a string corresponding to the specific dataset, e.g. 'human_augmented'
        # For the constraints model, 'datasets' is a list of strings corresponding to all the datasets to be included, e.g. ['human_augmented']
        footer += '_human_aug'

    with open(output_dir + f'/nl_goals_constraints_test{footer}.json', 'r', encoding='utf8') as f:
        test_data = json.load(f)
    if os.path.exists(output_dir + f'/nl_goals_constraints_kfold.json'):
        with open(output_dir + f'/nl_goals_constraints_kfold.json', 'r', encoding='utf8') as f:
            data = json.load(f)
    else:
        print("Creating kfold dataset")
        filename = f'consolidated_data{footer}.xlsx'
        data = read_data(filename, output_dir)
        with open(output_dir + f'/nl_goals_constraints_kfold.json', 'w', encoding='utf8') as f:
            json.dump(data, f, default=convert, indent=2)
    num_samples = len(data)
    train_data = data[:math.ceil(float(num_samples) * fold * 0.1)] + data[min(math.ceil(float(num_samples) * fold * 0.1) + math.ceil(float(num_samples) * 0.1), num_samples):]
    val_data = data[math.ceil(float(num_samples) * fold * 0.1): min(math.ceil(float(num_samples) * fold * 0.1) + math.ceil(float(num_samples) * 0.1), num_samples)]

    if augment:
        ### TODO Add test to make sure no datapoints are shared between augmented train and unaugmented val for each kfold split
        with open(output_dir + f'/nl_goals_constraints_kfold.json', 'r', encoding='utf8') as f:
            val_data_set = json.load(f)
        with open(output_dir + f'/nl_goals_constraints_train_human_kfold_aug.json', 'r', encoding='utf8') as f:
            train_aug_data = json.load(f)
        print("Len of val", len(val_data_set))
        print("Len of train", len(train_aug_data))
        num_samples_train = len(train_aug_data)
        num_samples_val =  len(val_data_set)
        val_start = math.ceil(float(num_samples_val) * fold * 0.1)
        val_end = min(math.ceil(float(num_samples_val) * fold * 0.1) + math.ceil(float(num_samples_val) * 0.1),
                  num_samples_val)
        train_data = train_aug_data[:2*val_start] + train_aug_data[2*val_end:]

        print("Fold : ", fold)
        print("len of train data : ", len(train_data))
        val_data = val_data_set[val_start: val_end]
        print("len of val_data : ", len(val_data))

        test_duplicate_values(train_data, val_data)
        #print("Len of val", len(val_data), "Val index : ", math.ceil(float(num_samples_val) * fold * 0.1), " : ", min(math.ceil(float(num_samples_val) * fold * 0.1) + math.ceil(float(num_samples_val) * 0.1), num_samples_val) )
        #print("Len of train", len(train_data), "Train Index : " , math.ceil(float(num_samples_train) * fold * 0.1)," : ",min(math.ceil(float(num_samples_train) * fold * 0.1) + math.ceil(float(num_samples_train) * 0.1), num_samples_train))

    with open(output_dir + f'/nl_goals_constraints_train{footer}_kfold.json', 'w', encoding='utf8') as f:
        json.dump(train_data, f, default=convert, indent=4)
    with open(output_dir + f'/nl_goals_constraints_val{footer}_kfold.json', 'w', encoding='utf8') as f:
        json.dump(val_data, f, default=convert, indent=4)
    with open(output_dir + f'/nl_goals_constraints_test{footer}_kfold.json', 'w', encoding='utf8') as f:
        json.dump(test_data, f, default=convert, indent=4)


def remove_null_values(train_data):
    """
    Deprecated function
    This function was used to remove null values from training dataset which were added when no valid augmentation was found
    Use the file - nl_goals_constraints_train_human_aug_with_null if you want to use this function
    """
    for strat in train_data:
        if strat['Text'] == "":
            train_data.remove(strat)

    return train_data
