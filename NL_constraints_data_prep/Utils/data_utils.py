import sys
import csv
import glob
import json
import logging
import xlsxwriter
from collections import defaultdict
import os
from typing import List, Optional
import math
from dataclasses import dataclass
import random
import torch
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
import tqdm
from enum import Enum
from typing import List, Optional
import pandas as pd

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def select_field(features, field):
	ans = []
	for feature in features:
		ans.append(getattr(feature, field))
	return ans

class InputExample(object):
	"""A single training/test example for multiple choice"""

	def __init__(self, text, goals, constraints, map):
		"""Constructs a InputExample.
		Args:
			text: Natural Language description of strategy.
			goals: Ratings for high-level goals associated with a strategy.
			constraints: Constraints associated with the RISK strategy
		"""
		self.text = text
		self.goals = goals
		self.constraints = constraints
		self.map = map

@dataclass(frozen=True)
class InputFeatures:
	"""
	A single set of features of data.
	Property names are the same names as the corresponding inputs to a model.
	"""

	example_id: str
	input_ids: List[List[int]]
	attention_mask: Optional[List[List[int]]]
	token_type_ids: Optional[List[List[int]]]
	constraints: List[List[int]]
	goals: List[float]


class NLConstraintsDataset(Dataset):
	features: List[InputFeatures]
	def __init__(
			self,
			model_args, 
			data_args,
			tokenizer: PreTrainedTokenizer,
			mode: Split = Split.train,
		):
			task = 'nlc'
			processor = processors['nlc']()
			cached_mode = mode
			
			cached_features_file = os.path.join(
				data_args.data_dir,
				"cached_{}_{}_{}_{}".format(
					cached_mode,
					list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
					str(data_args.max_seq_length),
					str(task),
				),
			)
			if os.path.exists(cached_features_file) and not data_args.overwrite_cache:
				self.features = torch.load(cached_features_file)
			else:
				load_data(data_args.data_dir)
				data_prep()
				print("Creating features from dataset file at %s", data_args.pickle_data_dir)
				print(mode)
				if mode == Split.dev:
					examples = processor.get_dev_examples(data_args.pickle_data_dir)
				elif mode == Split.test:
					examples = processor.get_test_examples(data_args.pickle_data_dir)
				else:
					examples = processor.get_train_examples(data_args.pickle_data_dir)
				# logger.info("Training number: %s", str(len(examples)))
				print("Training number: %s", str(len(examples)))
				self.features = convert_examples_to_features(
					examples,
					data_args.max_seq_length,
					tokenizer,
					pad_on_left=bool(model_args.model_name_or_path in ["xlnet"]),  # pad on the left for xlnet
					pad_token_segment_id=4 if model_args.model_name_or_path in ["xlnet"] else 0,
					model_type=model_args.model_name_or_path
				)
				# if args.local_rank in [-1, 0]:
				# logger.info("Saving features into cached file %s", cached_features_file)
				print("Saving features into cached file %s", cached_features_file)
				torch.save(self.features, cached_features_file)


			all_input_ids = torch.tensor(select_field(self.features, "input_ids"), dtype=torch.long)
			all_input_mask = torch.tensor(select_field(self.features, "attention_mask"), dtype=torch.long)
			all_segment_ids = torch.tensor(np.array(select_field(self.features, "token_type_ids"), dtype =float))
			all_goals_ids = torch.tensor(select_field(self.features, "goals"), dtype=torch.long)
			all_constraint_ids = torch.tensor(select_field(self.features, "constraints"), dtype=torch.long)
			print(all_input_ids.shape)
			print(all_goals_ids.shape)
			print(all_constraint_ids.shape)

			self.dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_goals_ids, all_constraint_ids)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i) -> InputFeatures:
		return self.dataset[i]



class DataProcessor(object):
	"""Base class for data converters for multiple choice data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the test set."""
		raise NotImplementedError()


class NLConstraintsProcessor(DataProcessor):
	""" Data Loader class to get Natural Language and Constraint data"""

	def get_train_examples(self, data_dir):
		# with open(data_dir, 'rb') as data_file:
		# 	data = pickle.load(data_file)
		return self._create_examples(data_dir + 'nl_goals_constraints_train.pkl')

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		return self._create_examples(data_dir + 'nl_goals_constraints_val.pkl')

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the test set."""
		return self._create_examples(data_dir + 'nl_goals_constraints_test.pkl')

	def _create_examples(self, data_dir: str, flag = True):
		print(data_dir)
		with open(data_dir, 'rb') as data_file:
			data = pickle.load(data_file)
		examples = []
		for point in data: 
			examples.append(InputExample(
				text = point['Text'],
				goals = point['Goals'],
				constraints = point['Constraints'],
				map = point['Map']))
		return examples
		

def load_data(data_dir):
	dataset = []

	def drop_prefix(self, prefix):
		self.columns = self.columns.str.lstrip(prefix)
		return self

	def drop_suffix(self, prefix):
		self.columns = self.columns.str.rstrip(prefix)
		return self

	pd.core.frame.DataFrame.drop_prefix = drop_prefix
	pd.core.frame.DataFrame.drop_suffix = drop_suffix

	for subdir, dirs, files in os.walk(data_dir):
		line_count = 0
		row = 1
		for filename in files:
			print(filename)
			if filename.endswith(".xlsx"):
				df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
				count = 0
				M1 = ['Q31','A_select', 'A_quantity', 'A_constraint', 'M1_G']
				M2 = ['Q73','B_select', 'B_quantity', 'B_constraint', 'M2_G']
				M3 = ['Q40','C_select', 'C_quantity', 'C_constraint', 'M3_G']
				M4 = ['Q45','D_select', 'D_quantity', 'D_constraint', 'M4_G']
				M5 = ['Q50','E_select', 'E_quantity', 'E_constraint', 'M5_G']
				m1_df = df.loc[:, df.columns.str.contains('|'.join(M1))].drop(index = 0)
				m1_df = m1_df.dropna(axis = 0, how = 'all')
				m1_df.insert(0, "M1_map", [1] * len(m1_df))
				m1_df = m1_df.rename(columns = {'Q31': 'M1_Instructions'})
				m2_df = df.loc[:, df.columns.str.contains('|'.join(M2))].drop(index = 0)
				m2_df = m2_df.dropna(axis = 0, how = 'all')
				m2_df.insert(0, "M2_map", [2] * len(m2_df))
				m2_df = m2_df.rename(columns = {'Q73': 'M2_Instructions'})
				m3_df = df.loc[:, df.columns.str.contains('|'.join(M3))].drop(index = 0)
				m3_df = m3_df.dropna(axis = 0, how = 'all')
				m3_df.insert(0, "M3_map", [3] * len(m3_df))
				m3_df = m3_df.rename(columns = {'Q40': 'M3_Instructions'})
				m4_df = df.loc[:, df.columns.str.contains('|'.join(M4))].drop(index = 0)
				m4_df = m4_df.dropna(axis = 0, how = 'all')
				m4_df.insert(0, "M4_map", [4] * len(m4_df))
				m4_df = m4_df.rename(columns = {'Q45': 'M4_Instructions'})
				m5_df = df.loc[:, df.columns.str.contains('|'.join(M5))].drop(index = 0)

				m5_df = m5_df.dropna(axis = 0, how = 'all')
				m5_df.insert(0, "M5_map", [5] * len(m5_df))
				m5_df = m5_df.rename(columns = {'Q50': 'M5_Instructions'})
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

				final = m1_df.append(m2_df, ignore_index = True).append(m3_df, ignore_index = True).append(m4_df, ignore_index = True).append(m5_df, ignore_index = True)
				writer = pd.ExcelWriter('../Output_Data/consolidated_data.xlsx', engine='xlsxwriter')
				final.to_excel(writer, sheet_name = 'Sheet1', index = False)
				writer.save()

def data_prep(output_dir = './Output_Data'):
	dataset = []
	constraint_types = {"I must have troops on ": 1, "I must not have troops on ": 2, "I must be able to access ": 3, "I need to protect the borders of ": 4, "I need a total of at least ": 5, "I must have at least ": 6, "I must have troops on at least ": 7, "I must place at least ": 8, "I must have troops on at most ": 9}
	value_names = {1: 'continent', 2: 'continent', 3: 'continent', 4: 'continent', 5: 'troops', 6: 'num_countries', 7: 'num_continents', 8: 'troops', 9: 'num_continents'}
	value_types = {
			'continent': {'Blue': 0, 'Green': 1, 'Yellow': 2, 'Red': 3, 'Purple': 4}, 
			'troops': {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, '11':10, '12':11, '13': 12, '14':13}, 
			'num_countries': {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6}, 
			'num_continents': {'1':0, '2':1, '3':2, '4':3, '5':4}
			}
	for subdir, dirs, files in os.walk(output_dir):
		line_count = 0
		row = 1
		for filename in files:
			if filename.endswith(".xlsx"):
				df = pd.ExcelFile(subdir + '/' + filename).parse('Sheet1')
				count = 0
				data = []
				print(df.shape[0])
				for j in range(df.shape[0]):
					print(j)
					goals = []
					text = df.iloc[j]['Instructions']
					if text == "" or not text == text:
						continue
					for i in range(1,7):
						goal_num = 'G' + str(i) + '_1'
						goals.append(df.iloc[j][goal_num])
					selections = []
					empty_flag = False
					for i in range(1,8):
						select_num = 'select_' + str(i)
						# print(str(df.iloc[j][select_num]))
						# print(type(df.iloc[j][select_num]))
						if isinstance(df.iloc[j][select_num], float):
							empty_flag = True
							break
						if 'select' not in df.iloc[j][select_num]:
							selections.append((df.iloc[j][select_num], df.iloc[j]['quantity_' + str(i)]))
					if empty_flag:
						continue
					constraints = []
					for i in range(1,9):
						constraint_num = 'constraint_' + str(i)
						constraint_text = df.iloc[j][constraint_num]
						if type(constraint_text) == float:
							continue
						max_len = -100
						max_ty = None
						# print(constraint_text)
						for ty in constraint_types.keys():
							if constraint_text.startswith(ty):
								if len(ty.split()) > max_len:
									max_len = max(len(ty.split()), max_len)
									max_ty = ty
						ty = max_ty
						if max_ty:
							value = constraint_text.split(ty)[1].split(' ')[0]
							constraints.append((constraint_types[ty], value_types[value_names[constraint_types[ty]]][value]))
					# print(constraints)
					for l in range(len(constraints), 9):
						constraints.append((-1,-1))
					data_dict = {'Map': df.iloc[j]['map'], 'Selections': selections, 'Text':text, 'Goals': goals, 'Constraints': constraints}
					data.append(data_dict)
	random.shuffle(data)
	# print(data)
	train_data = data[:math.ceil(float(len(data))*0.8)]
	val_data = data[math.ceil(float(len(data))*0.8):math.ceil(float(len(data))*0.9)]
	test_data = data[math.ceil(float(len(data))*0.9):]
	with open(output_dir + '/nl_goals_constraints_train.pkl', 'wb') as f:	
		pickle.dump(train_data, f)
	with open(output_dir + '/nl_goals_constraints_val.pkl', 'wb') as f:	
		pickle.dump(val_data, f)
	with open(output_dir + '/nl_goals_constraints_test.pkl', 'wb') as f:	
		pickle.dump(test_data, f)


def convert_examples_to_features(
	examples: List[InputExample],
	max_length: int,
	tokenizer: PreTrainedTokenizer,
	pad_token_segment_id=0,
	pad_on_left=False,
	pad_token=0,
	mask_padding_with_zero=True,
	model_type = 'gpt2'
	) -> List[InputFeatures]:
	"""
	Loads a data file into a list of `InputFeatures`
	"""

	# label_map = {label: i for i, label in enumerate(label_list)}

	features = []
	max_len = 0
	for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))
		choices_features = []
		text = example.text
		if model_type == 'gpt2':
			text += ' <CLS>'

		inputs = tokenizer(text, add_special_tokens = True, max_length = max_length, padding = "max_length", truncation = True, return_overflowing_tokens = True)
		if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
			logger.info(
				"Attention! you are cropping tokens (swag task is ok). "
				"you need to try to use a bigger max seq length!"
			)
		
		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		if model_type == 'gpt2':
			if not inputs["input_ids"] == tokenizer.pad_token_id:
				inputs["input_ids"][-1] = tokenizer.cls_token_id
		input_ids = inputs["input_ids"]
		attention_mask = (
					inputs["attention_mask"]  if "attention_mask" in inputs else None
				)
		token_type_ids = (
					inputs["token_type_ids"] if "token_type_ids" in inputs else None
				)
		
		# Zero-pad up to the sequence length.
		cls_token_location = -1
		cls_token_location = input_ids.index(tokenizer.cls_token_id)
		# choices_features.append((input_ids, attention_mask, token_type_ids, cls_token_location))



		# features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))
		features.append(
			InputFeatures(
				example_id=ex_index,
				input_ids=np.array(input_ids),
				attention_mask=np.array(attention_mask),
				token_type_ids=token_type_ids,
				constraints = example.constraints,
				goals = np.array(example.goals)
			)
		)
		for f in features[:2]:
			logger.info("*** Example ***")
			logger.info("feature: %s" % f)
	return features
processors = {"nlc": NLConstraintsProcessor}