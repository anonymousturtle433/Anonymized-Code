import os
import math
import random
from .data_utils_helpers_no_empty_constraints import *
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
import json
import pandas as pd


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


def setup_datasets(overwrite_cache=False, output_dir='/Output_Data/', vocab=None, include_unk=False, datasets = []):
    if overwrite_cache:
        load_data(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Data', datasets)
        data_prep()
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + output_dir
    train_json_path = output_dir + 'nl_goals_constraints_train.json'
    test_json_path = output_dir + 'nl_goals_constraints_test.json'
    val_json_path = output_dir + 'nl_goals_constraints_val.json'

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_json_path))
        vocab = build_vocab_from_iterator(json_iterator(train_json_path))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = create_data_from_iterator(
        vocab, json_iterator(train_json_path, yield_cls=True), include_unk)
    logging.info('Creating validation data')
    val_data, val_labels = create_data_from_iterator(
        vocab, json_iterator(val_json_path, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = create_data_from_iterator(
        vocab, json_iterator(test_json_path, yield_cls=True), include_unk)
    # if len(train_labels ^ test_labels) > 0:
    # 	raise ValueError("Training and test labels don't match")
    return (NLConstraintsDataset(vocab, train_data, train_labels),
            NLConstraintsDataset(vocab, val_data, val_labels),
            NLConstraintsDataset(vocab, test_data, test_labels))





def load_data(output_dir, datasets = []):

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
            elif filename.endswith(".csv"):
                if "Synthetic" in filename and "syn" in datasets:
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
        writer = pd.ExcelWriter(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Output_Data/consolidated_data.xlsx',
                                engine='xlsxwriter')
        final.to_excel(writer, sheet_name='Sheet1', index=False, encoding = 'utf8')
        writer.save()


def data_prep(output_dir='/Output_Data'):
    dataset = []

    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + output_dir
    # print(output_dir)
    data = []
    for subdir, dirs, files in os.walk(output_dir):
        line_count = 0
        row = 1
        for filename in files:
            if filename.endswith(".xlsx") and "OLD" not in filename:
                df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
                count = 0
                for j in range(df.shape[0]):
                    # print(j)
                    goals = []
                    text = df.iloc[j]['Instructions']
                    if text == "" or not text == text:
                        continue
                    for i in range(1, 7):
                        goal_num = 'G' + str(i) + '_1'
                        goals.append(int(df.iloc[j][goal_num]))
                    selections = []
                    selections.append(('<sos>', '<sos>'))
                    empty_flag = False
                    for i in range(1, 8):
                        select_num = 'select_' + str(i)
                        if isinstance(df.iloc[j][select_num], float):
                            empty_flag = True
                            break
                        if "Synthetic" in df.iloc[j][select_num]:
                            selections.append(('<pad>', '<pad>'))
                            continue
                        if 'select' not in df.iloc[j][select_num]:
                            selections.append((df.iloc[j][select_num], int(df.iloc[j]['quantity_' + str(i)])))
                    selections.append(('<eos>', '<eos>'))
                    for l in range(len(selections), 9):
                        selections.append(('<pad>', '<pad>'))
                    if empty_flag:
                        continue
                    constraints = []
                    for i in range(1, 9):
                        constraint_num = 'constraint_' + str(i)
                        constraint_text = df.iloc[j][constraint_num]
                        if type(constraint_text) == float:
                            continue
                        max_len = -100
                        max_ty = None
                        for ty in constraint_types.keys():
                            if constraint_text.startswith(ty):
                                if len(ty.split()) > max_len:
                                    max_len = max(len(ty.split()), max_len)
                                    max_ty = ty
                        ty = max_ty
                        if max_ty:
                            value = constraint_text.split(ty)[1].split(' ')[0]
                            # constraints.append(
                            #     (constraint_types[ty], value_types[value_names[constraint_types[ty]]][value]))
                            # Convert constraint pair into a single digit
                            # constraints.append(constraint_types[ty] * len(VALUE_TYPES) + value_types[value_names[constraint_types[ty]]][value])
                            constraints.append(encode_constraint_no_empty(ty, value))
                    for l in range(len(constraints), 8):
                        # constraints.append((0, 0))
                        constraints.append(0)
                    data_dict = {'Map': df.iloc[j]['map'], 'Selections': selections, 'Text': text, 'Goals': goals,
                                 'Constraints': constraints}
                    data.append(data_dict)
    # random.shuffle(data)
    train_data = data[:math.ceil(float(len(data)) * 0.8)]
    val_data = data[math.ceil(float(len(data)) * 0.8):math.ceil(float(len(data)) * 0.9)]
    test_data = data[math.ceil(float(len(data)) * 0.9):]
    with open(output_dir + '/nl_goals_constraints_train.json', 'w', encoding='utf8') as f:
        json.dump(train_data, f, default=convert)
    with open(output_dir + '/nl_goals_constraints_val.json', 'w', encoding='utf8') as f:
        json.dump(val_data, f, default=convert)
    with open(output_dir + '/nl_goals_constraints_test.json', 'w', encoding='utf8') as f:
        json.dump(test_data, f, default=convert)
# with open(output_dir + '/nl_goals_constraints_train.pkl', 'wb') as f:
# 	pickle.dump(train_data, f)
# with open(output_dir + '/nl_goals_constraints_val.pkl', 'wb') as f:
# 	pickle.dump(val_data, f)
# with open(output_dir + '/nl_goals_constraints_test.pkl', 'wb') as f:
# 	pickle.dump(test_data, f)


# load_data('./Data')
# data_prep()
# train, val, test = setup_datasets('./Output_Data/')
# vocab = train.get_vocab()
# country_map = {"Yellow_A": 0, "Yellow_B": 1, "Yellow_C": 2, "Yellow_D": 3, "Blue_A": 4, "Blue_B": 5, "Blue_C": 6, "Blue_D": 7, "Green_A": 8, "Green_B": 9, "Green_C": 10, "Green_D": 11, "Green_E": 12, "Purple_A": 13, "Purple_B": 14, "Purple_B": 15, "Purple_C": 16, "Purple_D": 17, "Purple_E": 18, "Red_A": 19, "Red_B": 20, "Red_C": 21}
# def collate_fn(batch):
#     texts, constraints, selections, goals, maps = [], [], [], [], []
#     for selection, constraint, goal, map, txt in batch:
#         texts.append(txt)
#         for s in selection:
#         	if not s[0] == -1:
#         		s[0] = country_map[s[0]]
#         selections.append(torch.tensor(selection))
#         constraint = torch.tensor(constraint)
#         constraints.append(constraint)
#         goals.append(torch.tensor(goal))
#         maps.append(map)
#     texts = pad_sequence(texts, batch_first = True, padding_value = vocab.stoi['<pad>'])
#     goals = torch.stack(goals)
#     maps = torch.tensor(maps)
#     constraints = torch.stack(constraints)
#     selections = torch.stack(selections)
#     return texts, constraints, goals, maps, selections

# dataloader = DataLoader(train, batch_size=8, collate_fn=collate_fn)

# # for idx, (texts, constraints, goals, maps, selections) in enumerate(dataloader):
# # 	# print(idx, texts, constraints, goals, maps, selections)
# # 	print(texts.shape)
# # 	print(goals.shape)
# # 	print(maps.shape)
# # 	print(selections.shape)
# # 	print(constraints.shape)

# # print(train.get_vocab().stoi)
# # sdfsdf

# processors = {"nlc": NLConstraintsProcessor}
