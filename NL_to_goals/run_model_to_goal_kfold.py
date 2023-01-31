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
import pickle
import os
from functools import partial

# sys.path.append('../NL_constraints_data_prep/Utils')
from NL_constraints_data_prep.Utils.Dataset import setup_datasets, prepare_data
from NL_constraints_data_prep.Utils.data_utils_helpers import decode_batch

from NL_to_goals.models.test_rnn_to_goal import Encoder, Attention, Regressor, Classifier, Seq2GoalClass, Seq2GoalValue
from NL_to_goals.models.test_bert_to_goal import BertToGoalValue, BertToGoalClass, GoalLabelSmoothingLoss
from NL_to_goals.models.test_bert_to_goal import Classifier as BertClassifier
from NL_to_goals.models.test_bert_to_goal import Regressor as BertRegressor
from NL_to_goals.models.test_bert_to_goal import SelectionsEncoder as BertSelectionsEncoder

from NL_to_goals.Utils.training_utils import *
from NL_to_goals.Utils.evaluation_utils import *
from NL_to_goals.Utils.constants import *

"""
python script for running kfold tests for a particular configuration. 
You should run hyperparamter search using run_model_to_goal file and then a kfold test for the model with the best 
configurations.

"""

# arguments which can be passed to the script. See help statements for more detail on each argument
parser = argparse.ArgumentParser()
parser.add_argument("--warmup_steps", default=10, type=int, help="Number of warmup_steps for linear step scheduler.")
# parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets") #unused
parser.add_argument("--gpu_core", default=0, type=int, help="Which GPU to use when on MultiGpu systems")
parser.add_argument("--save_path", default='model_results/', type=str, help="Root directory for saving results")
parser.add_argument("--num_goal_buckets", default=5, type=int, help="Number of classes to predict "
                                                                    "within goal value range")
parser.add_argument("--num_goals", default=6, type=int, help="Number of goals in data")
parser.add_argument("--goal_value_range", default=100, type=int, help="Max value of goal data; "
                                                                      "assumed that min value is the negative")
parser.add_argument("--reduced_num_goal_buckets", default=0, type=int, help="Smaller number of classification targets, "
                                                                            "for use in evaluation section. Only works "
                                                                            "with regression models. Set to 0 to disable")
parser.add_argument("--pretrained_path", default='pretrained_models', type=str,
                    help="Directory in which saved pretrained models are")
parser.add_argument('--save_pretrained_models', dest='save_pretrained_models', action='store_true')
parser.set_defaults(save_pretrained_models=False)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_core}' if torch.cuda.is_available() else 'cpu')

evaluation_bucket_nums = (args.num_goal_buckets, args.reduced_num_goal_buckets)
metrics = METRIC_NAMES  # need to add way for user to specify which metrics they want to record

base_output_dir = DATASET_BASE_PATH

if __name__ == "__main__":
    # define possible loss functions
    cross_entropy_criterion = nn.CrossEntropyLoss()
    MSE_criterion = nn.MSELoss()
    L1_criterion = nn.L1Loss()
    combined_loss = (cross_entropy_criterion, MSE_criterion)
    ordinal_loss = torch.nn.BCELoss()
    # label_smoohing amounts were arbitraily selected and can be altered
    label_smoothing_loss_0_05 = GoalLabelSmoothingLoss(0.05, args.num_goals, args.num_goal_buckets).to(device)
    label_smoothing_loss_0_1 = GoalLabelSmoothingLoss(0.1, args.num_goals, args.num_goal_buckets).to(device)
    label_smoothing_loss_0_3 = GoalLabelSmoothingLoss(0.3, args.num_goals, args.num_goal_buckets).to(device)

    if args.num_goal_buckets == 5:
        decay_rate = 0.3
        acc_dist_threshold = 2
    elif args.num_goal_buckets == 3:
        decay_rate = 0.1
        acc_dist_threshold = 1
    elif args.num_goal_buckets == 10:
        decay_rate = 0.3
        acc_dist_threshold = 3
    else:
        decay_rate = 0.3
        acc_dist_threshold = 2

    # associate loss functions with a string name
    conf_names = {cross_entropy_criterion: 'ce',
                  MSE_criterion: "MSE",
                  L1_criterion: "L1",
                  combined_loss: "comb",
                  label_smoothing_loss_0_05: 'ls_0.05',
                  label_smoothing_loss_0_1: 'ls_0.1',
                  label_smoothing_loss_0_3: 'ls_0.3',
                  ordinal_loss: 'ord'}

    # define which types of models you want to evaluate
    architecture_search_space = ['bert']  # rnn, bert, transformer

    # define the datasets which would want to train models on
    #   dataset_search_space = ['h', 's', 'sa', 'ha']
    dataset_search_space = ['h', 'ha']

    # define hyperparameter search spaces for each architecture here
    # if list is only None value, this parameter is not used for this architecture - leave as None
    # rnn hyperparameter space
    # rnn_search_space = {
    #   "lr": [1e-3,5e-3],
    #   "batch_size": [4],
    #   "weight_decay": [0],
    #   "num_train_epochs": [1],
    #   "optimizer": ['Adam'],
    #   "enc_embedding": [32],
    #   "dropout": [0.5],
    #   "hidden_size": [32],
    #   "attention": [32],
    #   "criterion": [cross_entropy_criterion],
    #   'num_attention_heads': [None],
    #   'num_hidden_layers': [None],
    #   'pretrained_names':[None],
    #   'selection_type':[None],
    #   'selection_output_size': [768],
    #   'use_text_encoder': [True, False], #whether text encoder is enabled or disabled
    #   'goal_tokens': [None]
    # }
    #
    # # transformer hyperparameter space
    # transformer_search_space = {
    #   "lr": [1e-2, 1e-3],
    #   "batch_size": [8],
    #   "weight_decay": [0],
    #   "num_train_epochs": [25],
    #   "optimizer": ['Adam'],
    #   "enc_embedding": [None],
    #   "dropout": [0.5],
    #   "hidden_size": [None],
    #   "attention": [None],
    #   "criterion": [cross_entropy_criterion],
    #   'num_attention_heads': [8],
    #   'num_hidden_layers': [8],
    #   'pretrained_names':[None],
    #   'selection_type': [None],
    #   'selection_output_size': [168,768],
    #   'use_text_encoder': [False, True],
    #   'goal_tokens': [None]
    # }

    # 5 bucket bert search space
    # bert_search_space = {
    #     "lr": [1e-4, 1e-5],
    #     "batch_size": [8],
    #     "weight_decay": [0.1],
    #     "num_train_epochs": [25],
    #     "optimizer": ['Adam'],
    #     "enc_embedding": [None],
    #     "dropout": [0.5],
    #     "hidden_size": [None],
    #     "attention": [None],
    #     "criterion": [combined_loss],
    #     'num_attention_heads': [None],
    #     'num_hidden_layers': [None],
    #     'pretrained_names':[None,
    #                         'd-sa_a-bert_lr-0.0001_e-2_g-True.pt',
    #                         'd-s_a-bert_lr-0.0001_e-2_g-True.pt',
    #                         'd-sa_a-bert_lr-5e-05_e-2_g-True.pt',
    #                         'd-s_a-bert_lr-5e-05_e-2_g-True.pt'], #name of pretrained models to use; if not None, will checked specified folder
    #     'selection_type': ['text_selections_no_map'],
    #     'selection_output_size': [32],
    #     'use_text_encoder': [True],
    #     'goal_tokens': [True], # whether or not to add per goal classification tokens to the bert model
    #     'uniform_buckets':[True],
    #     'mse_weight':[(0,1), (0.2,0.8)], #(initial weight, final weight) for the MSE metric when using combined loss
    #     'coral':[False],
    #     'mse_annealing': ['linear', 'logistic']
    #   }

    rnn_search_space = {
        "lr": [1e-3, 5e-3],
        "batch_size": [4, 8],
        "weight_decay": [0, 0.1, 0.001],
        "num_train_epochs": [25],
        "optimizer": ['Adam'],
        "enc_embedding": [32],
        "dropout": [0.5],
        "hidden_size": [32],
        "attention": [32],
        "criterion": [cross_entropy_criterion],
        'num_attention_heads': [None],
        'num_hidden_layers': [None],
        'pretrained_names': [None],
        'selection_type': [None],
        'selection_output_size': [768],
        'use_text_encoder': [True, False],  # whether text encoder is enabled or disabled
        'goal_tokens': [None],
        'uniform_buckets': [True],
        'mse_weight': [(0, 1)],  # (initial weight, final weight) for the MSE metric when using combined loss
        'coral': [False],
        'mse_annealing': ['linear']
    }

    # 3 bucket bert search space
    bert_search_space = {
        "lr": [1e-4, 1e-5],
        "batch_size": [8],
        "weight_decay": [0.1],
        "num_train_epochs": [25],
        "optimizer": ['Adam'],
        "enc_embedding": [None],
        "dropout": [0.5],
        "hidden_size": [None],
        "attention": [None],
        "criterion": [combined_loss],
        'num_attention_heads': [None],
        'num_hidden_layers': [None],
        'pretrained_names': [None,
                             'd-sa_a-bert_lr-0.0001_e-2_g-True_c-False.pt',
                             'd-s_a-bert_lr-0.0001_e-2_g-True_c-False.pt',
                             'd-sa_a-bert_lr-1e-05_e-2_g-True_c-False.pt',
                             'd-s_a-bert_lr-1e-05_e-2_g-True_c-False.pt'],
        # name of pretrained models to use; if not None, will checked specified folder
        'selection_type': ['text_selections_no_map'],
        'selection_output_size': [32],
        'use_text_encoder': [True],
        'goal_tokens': [True],  # whether or not to add per goal classification tokens to the bert model
        'uniform_buckets': [True],
        'mse_weight': [(0, 1), (0.2, 0.8)],
        # (initial weight, final weight) for the MSE metric when using combined loss
        'coral': [False],
        'mse_annealing': ['linear', 'logistic']
    }

    # tracks the best performing model as measured by cumulative acc over all configs
    # while it would be preferable to simply save all the models and analyze the performance serperately, that would
    # consume far too much storage space
    # cumulative validation acc, model name, epoch saved
    best_acc_and_model = [0.0, "", 0]

    # iterate through architecture choices at the lowest level, since they require different dataprep and setup
    for dataset_name in dataset_search_space:
        datasets = DATASET_CODE_TO_NAME[dataset_name]
        train_accs = []
        valid_accs = []
        best_valid_acc = float("-inf")
        best_kfold_acc = 0
        for fold in range(10):
            prepare_data(datasets, kfold=True, fold=fold)
            for arch_name in architecture_search_space:
                # define path for saving training data
                save_path = args.save_path + f"/{dataset_name}_{arch_name}"

                # depending on the current architecture name, perform dataprep and other setup steps
                if arch_name == 'rnn':
                    # if args.overwrite_cache is True, will generate json data files from raw XL files in Data folder (if data present)
                    # if false, will look for json data files in Output_data_folder (in ...data_prep/utils)
                    # train_dataset, valid_dataset, test_dataset = setup_datasets(args.overwrite_cache, output_dir=os.path.join(base_output_dir, dataset_name, ""))
                    train_dataset, valid_dataset, test_dataset = setup_datasets(datasets, include_unk=True, kfold = True)
                    vocab = train_dataset.get_vocab()
                    initialize_variables(
                        vocab)  # function from training utils, required for collate function to work properly.

                    #         src_vocab = vocab.stoi
                    src_vocab = vocab.get_stoi()

                    # set correct tokenizer
                    current_tokenizer = None

                    # set search space variable to corresponding architecture search space
                    search_space = rnn_search_space

                elif arch_name == 'bert':
                    # set up for bert architecture

                    # collect bert model and tokenizer
                    model_name = "roberta-base"
                    bert_model = None  # explicitly clear old bert model
                    bert_model = AutoModel.from_pretrained(model_name)

                    model_config = bert_model.config
                    # set correct tokenizer
                    current_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    bert_vocab = current_tokenizer.get_vocab()

                    train_dataset, valid_dataset, test_dataset = setup_datasets(datasets, include_unk=True,
                                                                                tokenizer=current_tokenizer, kfold = True)

                    vocab = train_dataset.get_vocab()
                    #         src_vocab = vocab.stoi
                    src_vocab = vocab.get_stoi()

                    # set search space variable to corresponding architecture search space
                    search_space = bert_search_space

                elif arch_name == 'transformer':
                    # set up for transformer architecture

                    # path to training vocab. Need to evaluate where this training vocab came from, as it seems too short to be a full vocab
                    paths = ['./Utils/train_vocab_text.txt']
                    # Initialize a tokenizer
                    # set correct tokenizer variable
                    current_tokenizer = ByteLevelBPETokenizer()

                    # Customize training
                    current_tokenizer.train(files=paths, vocab_size=1500, min_frequency=1, special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                    ])
                    current_tokenizer._tokenizer.post_processor = BertProcessing(
                        ("</s>", current_tokenizer.token_to_id("</s>")),
                        ("<s>", current_tokenizer.token_to_id("<s>")),
                    )

                    # has to be 2 less than the positional embedding size, probably to handle the start and end tokens? I'm not entirely sure.
                    current_tokenizer.enable_truncation(510)
                    train_dataset, valid_dataset, test_dataset = setup_datasets(datasets, include_unk=True,
                                                                                tokenizer=current_tokenizer, kfold = True)
                    vocab = train_dataset.get_vocab()
                    #         src_vocab = vocab.stoi
                    src_vocab = vocab.get_stoi()

                    # set search space variable to corresponding architecture search space
                    search_space = transformer_search_space

                # search over hyperparameters for current architecture
                for num, conf in enumerate(itertools.product(search_space["criterion"],
                                                             search_space["lr"],
                                                             search_space["batch_size"],
                                                             search_space["weight_decay"],
                                                             search_space["num_train_epochs"],
                                                             search_space["optimizer"],
                                                             search_space["enc_embedding"],
                                                             search_space["dropout"],
                                                             search_space["hidden_size"],
                                                             search_space["attention"],
                                                             search_space["num_attention_heads"],
                                                             search_space["num_hidden_layers"],
                                                             search_space['pretrained_names'],
                                                             search_space['selection_type'],
                                                             search_space['use_text_encoder'],
                                                             search_space['selection_output_size'],
                                                             search_space['goal_tokens'],
                                                             search_space['uniform_buckets'],
                                                             search_space['mse_weight'],
                                                             search_space['coral'],
                                                             search_space['mse_annealing'])):
                    print("Training config: ", fold)
                    params = {
                        'input_dim': len(src_vocab),
                        'constraint_dim': len(CONSTRAINT_TYPES),
                        'value_dim': len(VALUE_TYPES),
                        'batch_size': conf[2],
                        'enc_emb_dim': conf[6],
                        'enc_hidden_dim': conf[8],
                        'value_hidden_dim': conf[8],
                        'enc_dropout': conf[7],
                        'clip': 1,
                        'optimizer': conf[5],
                        'lr': conf[1],
                        'min_lr': 0.0001,
                        'weight_decay': conf[3],
                        'num_epochs': conf[4],
                        'lr_decay': 0.95,
                        'attention': conf[9],
                        'scheduler': 'StepLR',
                        'warmup_steps': 50,
                        'criterion': conf[0],
                        'criterion_name': conf_names[conf[0]],
                        'num_attention_heads': conf[10],
                        'num_hidden_layers': conf[11],
                        'pretrained_model': conf[12],
                        'selection_type': conf[13],
                        'use_text_encoder': conf[14],
                        'selection_output_size': conf[15],
                        'goal_tokens': conf[16],
                        'uniform_buckets': conf[17],
                        'mse_weight': conf[18],
                        'coral': conf[19],
                        'mse_annealing': conf[20]
                    }
                    # skipping illegal configurations
                    if (params['selection_type'] == None or 'text' in params['selection_type']) and not params[
                        'use_text_encoder']:
                        print('Skipping parameter configuration as neither text nor selection encoders were enabled')
                        continue
                    if params['pretrained_model'] != None:
                        if 'g-True' in params['pretrained_model'] and not params['goal_tokens']:
                            print('Skipping parameter configuration because of mismatch between goal tokenization')
                            continue
                        elif 'g-False' in params['pretrained_model'] and params['goal_tokens']:
                            print('Skipping parameter configuration because of mismatch between goal tokenization ')
                            continue

                        if 'c-True' in params['pretrained_model'] and not params['coral']:
                            print('Skipping parameter configuration because of mismatch between coral')
                            continue
                        elif 'c-False' in params['pretrained_model'] and params['coral']:
                            print('Skipping parameter configuration because of mismatch between coral')
                            continue

                    if params['criterion_name'] == 'ce' and params['mse_weight'] != 1:
                        print('Skipping parameter configuration since varying mse weight while only using ce criterion')
                        continue

                    if params['criterion_name'] != 'ord' and params['coral']:
                        print(
                            'Skipping parameter configuration since we can not use coral without using an ordinal model')
                        continue

                    # define log file that can be used to idtenify model results. Only important model hyperparameters have been included
                    if arch_name == 'rnn':
                        log_file = f"run-{fold}," \
                                   f"data-{dataset_name}," \
                                   f"arch-{arch_name}," \
                                   f"crit-{params['criterion_name']}," \
                                   f"lr-{params['lr']}," \
                                   f"b_size-{params['batch_size']}," \
                                   f"epo-{params['num_epochs']}," \
                                   f"opt-{params['optimizer']}," \
                                   f"enc_emb-{params['enc_emb_dim']}," \
                                   f"drop-{params['enc_dropout']}," \
                                   f"hid-{params['enc_hidden_dim']}," \
                                   f"att-{params['attention']}," \
                                   f"buckets-{args.num_goal_buckets}," \
                                   f"pretrained-{params['pretrained_model']}," \
                                   f"sel-{params['selection_type']}," \
                                   f"sel_size-{params['selection_output_size']}," \
                                   f"text-{params['use_text_encoder']}"
                    elif arch_name == 'transformer':
                        log_file = f"run-{fold}," \
                                   f"data-{dataset_name}," \
                                   f"arch-{arch_name}," \
                                   f"crit-{params['criterion_name']}," \
                                   f"lr-{params['lr']}," \
                                   f"b_size-{params['batch_size']}," \
                                   f"epo-{params['num_epochs']}," \
                                   f"att-head-{params['num_attention_heads']}," \
                                   f"hid-layers-{params['num_hidden_layers']}," \
                                   f"drop-{params['enc_dropout']}," \
                                   f"buckets-{args.num_goal_buckets}," \
                                   f"pretrained-{params['pretrained_model']}," \
                                   f"sel-{params['selection_type']}," \
                                   f"sel_size-{params['selection_output_size']}," \
                                   f"text-{params['use_text_encoder']}," \
                                   f"g_tok-{params['goal_tokens']}"
                    elif arch_name == 'bert':
                        log_file = f"run-{fold}," \
                                   f"data-{dataset_name}," \
                                   f"arch-{arch_name}," \
                                   f"crit-{params['criterion_name']}," \
                                   f"lr-{params['lr']}," \
                                   f"epo-{params['num_epochs']}," \
                                   f"buckets-{args.num_goal_buckets}," \
                                   f"pretrained-{params['pretrained_model']}," \
                                   f"sel-{params['selection_type']}," \
                                   f"sel_size-{params['selection_output_size']}," \
                                   f"text-{params['use_text_encoder']}," \
                                   f"g_tok-{params['goal_tokens']}," \
                                   f"uniform-{params['uniform_buckets']}," \
                                   f"mse_w-{params['mse_weight'][0]}to{params['mse_weight'][1]}," \
                                   f"mse_a-{params['mse_annealing']}," \
                                   f"coral-{params['coral']}"

                    print(log_file)
                    writer = SummaryWriter(f'{save_path}/grid_search/' + log_file)

                    # set up dataloaders

                    if arch_name == 'bert' and params['goal_tokens']:  # add tokens for each goal if using that approach
                        current_tokenizer.add_tokens(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

                    if arch_name == 'bert':
                        bert_vocab = current_tokenizer.get_vocab()

                    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'],
                                                  collate_fn=partial(collate_fn, tokenizer=current_tokenizer,
                                                                     selection_type=params['selection_type'],
                                                                     goal_tokens=params['goal_tokens']), shuffle=True)
                    val_dataloader = DataLoader(valid_dataset, batch_size=params['batch_size'],
                                                collate_fn=partial(collate_fn, tokenizer=current_tokenizer,
                                                                   selection_type=params['selection_type'],
                                                                   goal_tokens=params['goal_tokens']))
                    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'],
                                                 collate_fn=partial(collate_fn, tokenizer=current_tokenizer,
                                                                    selection_type=params['selection_type'],
                                                                    goal_tokens=params['goal_tokens']))

                    # define constants
                    RANGE_LIMIT = args.goal_value_range
                    NUM_GOAL_BUCKETS = args.num_goal_buckets
                    CLIP = params['clip']
                    NUM_GOALS = args.num_goals

                    # from run bert/transformer
                    HIDDEN_STATE_SIZE = BERT_TO_HIDDEN_DIM[params['use_text_encoder']]
                    if params['selection_type'] != None and 'text' not in params['selection_type']:
                        HIDDEN_STATE_SIZE += params['selection_output_size']

                    DROPOUT = params['enc_dropout']
                    NUM_ATTENTION_HEADS = params['num_attention_heads']
                    NUM_HIDDEN_LAYERS = params['num_hidden_layers']

                    criterion = params['criterion']

                    # set up pytorch model for architecture and configuration
                    if arch_name == 'rnn':
                        enc = Encoder(params['input_dim'], params['enc_emb_dim'], params['enc_hidden_dim'],
                                      params['enc_dropout'])
                        attns = nn.ModuleList()
                        for i in range(NUM_GOALS):
                            attns.append(Attention(params['enc_hidden_dim'], params['attention']))

                        if isinstance(criterion, nn.CrossEntropyLoss) \
                                or type(criterion) == tuple \
                                or isinstance(criterion, GoalLabelSmoothingLoss):

                            classifier = Classifier(params['enc_hidden_dim'], NUM_GOALS, NUM_GOAL_BUCKETS, attns)
                            model = Seq2GoalClass(enc, classifier).to(device)
                            # initialize model weights
                            model.apply(init_weights)

                        elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                            reg = Regressor(params['enc_hidden_dim'], NUM_GOALS, attns)
                            model = Seq2GoalValue(enc, reg).to(device)
                        model.apply(init_weights)
                    elif arch_name == 'bert':
                        # define the encoders for selection and text
                        if params['use_text_encoder']:
                            model_name = 'roberta-base'
                            del bert_model  # explicitly clear old bert model
                            bert_model = AutoModel.from_pretrained(model_name)
                        else:
                            bert_model = None

                        if params['selection_type'] != None and 'text' not in params['selection_type']:
                            selection_encoder = BertSelectionsEncoder(
                                SELECTION_TYPE_TO_HIDDEN_DIM[params['selection_type']],
                                params['selection_output_size'],
                                DROPOUT)
                            selection_encoder.apply(init_weights)
                        else:
                            selection_encoder = None

                        if isinstance(criterion, nn.CrossEntropyLoss) \
                                or type(criterion) == tuple \
                                or isinstance(criterion, GoalLabelSmoothingLoss) \
                                or isinstance(criterion, nn.BCELoss):

                            if isinstance(criterion, nn.BCELoss):  # if using BCE loss, using an ordinal model
                                is_ordinal = True
                            else:
                                is_ordinal = False

                            classifier = BertClassifier(HIDDEN_STATE_SIZE, NUM_GOALS, NUM_GOAL_BUCKETS, DROPOUT,
                                                        is_ordinal=is_ordinal, is_coral=params['coral'])
                            # initialize model weights just for untrained head
                            classifier.apply(init_weights)
                            model = BertToGoalClass(bert_model, classifier, selection_encoder,
                                                    goal_tokens=params['goal_tokens']).to(device)

                        elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                            reg = BertRegressor(HIDDEN_STATE_SIZE, NUM_GOALS, DROPOUT)
                            # initialize model weights just for untrained head
                            reg.apply(init_weights)
                            model = BertToGoalValue(bert_model, reg).to(device)

                        if bert_model != None:
                            bert_model.resize_token_embeddings(
                                len(current_tokenizer))  # make sure that bert embedding is expanded if using goal tokens

                        if params['pretrained_model'] != None:
                            load_name = params['pretrained_model']

                            model_dict = model.state_dict()  # get current model parameters
                            # pretrained_model_dict = torch.load(f'{args.pretrained_path}/{load_name}.pt')
                            print(f'{args.pretrained_path}/{load_name}.pt')
                            pretrained_model_dict = torch.load(f'{args.pretrained_path}/{load_name}.pt')
                            #               if 'g-True' in load_name and params['goal_tokens']:
                            #                 bert_model.resize_token_embeddings(len(current_tokenizer)) #make sure that bert embedding is expanded before loading

                            if params['selection_type'] != None and 'text' not in params[
                                'selection_type']:  # if using selection encoder
                                # load all pretrained weights from encoder (bert), but not from head
                                pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if
                                                         k in model_dict and 'classifier' not in k}
                                model_dict.update(pretrained_model_dict)  #
                            else:
                                model_dict = pretrained_model_dict
                            model.load_state_dict(model_dict)
                            print('Weights loaded successfully')




                    elif arch_name == 'transformer':
                        vocab_size_from_tokenizer = current_tokenizer.get_vocab_size()
                        custom_config = RobertaConfig(num_attention_heads=NUM_ATTENTION_HEADS,
                                                      num_hidden_layers=NUM_HIDDEN_LAYERS,
                                                      vocab_size=vocab_size_from_tokenizer, type_vocab_size=2)
                        bert_model = RobertaModel(custom_config)

                        if isinstance(criterion, nn.CrossEntropyLoss) or \
                                type(criterion) == tuple or \
                                isinstance(criterion, GoalLabelSmoothingLoss) \
                                or isinstance(criterion, nn.BCELoss):

                            if isinstance(criterion, nn.BCELoss):  # if using BCE loss, using an ordinal model
                                is_ordinal = True
                            else:
                                is_ordinal = False

                            classifier = BertClassifier(HIDDEN_STATE_SIZE, NUM_GOALS, NUM_GOAL_BUCKETS, DROPOUT,
                                                        is_ordinal=is_ordinal)
                            model = BertToGoalClass(bert_model, classifier).to(device)
                            # initialize model weights
                            model.apply(init_weights)
                        elif isinstance(criterion, nn.L1Loss) or isinstance(criterion, nn.MSELoss):
                            reg = BertRegressor(HIDDEN_STATE_SIZE, NUM_GOALS, DROPOUT)
                            model = BertToGoalValue(bert_model, reg).to(device)
                            # initialize model weights
                            model.apply(init_weights)

                    t_total = len(train_dataloader) * params['num_epochs']

                    # intialize opimizer, same process for all architectures
                    if params['optimizer'] == 'Adam':
                        if arch_name == 'bert':
                            # For bert architecture, set different learning es for encoder/decoder
                            bert_params = []
                            non_bert_params = []
                            for name, param in model.named_parameters():
                                if 'bert_model' in name:
                                    bert_params.append(param)
                                else:
                                    # also includes selections encoder and decoder parameters
                                    non_bert_params.append(param)
                            optimizer = optim.AdamW(
                                [{"params": bert_params, "lr": 1e-5, "weight_decay": params['weight_decay']},
                                 {"params": non_bert_params, "lr": params['lr'], "weight_decay": params['weight_decay']}
                                 ]
                            )
                        else:
                            optimizer = optim.AdamW(
                                model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
                            )
                    elif params['optimizer'] == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

                    # initialize learning rate scheduler, same process for all architectures
                    if params['scheduler'] == 'StepLR':
                        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=params['warmup_steps'],
                                                                    num_training_steps=t_total)

                    # generate and log initial evaluation data
                    # uses initial MSE weighting
                    initial_mse_weight = params['mse_weight'][0]
                    final_mse_weight = params['mse_weight'][1]
                    print("Initialized params are : ", params)
                    training_metric_results, validation_metric_results = evaluation_manager(model,
                                                                                            (train_dataloader,
                                                                                             val_dataloader),
                                                                                            optimizer,
                                                                                            criterion,
                                                                                            device,
                                                                                            metrics,
                                                                                            range_limit=RANGE_LIMIT,
                                                                                            evaluation_bucket_nums=evaluation_bucket_nums,
                                                                                            uniform_buckets=params[
                                                                                                'uniform_buckets'],
                                                                                            mse_weight=initial_mse_weight,
                                                                                            decay_rate=decay_rate,
                                                                                            dist_threshold=acc_dist_threshold)

                    record_metrics(writer, metrics, evaluation_bucket_nums, training_metric_results,
                                   validation_metric_results, step=0)
                    print_metrics(0, training_metric_results, validation_metric_results)

                    best_acc_and_model = [0.0, "", 0]
                    # train model, save model if it surpasses previous models, and record model performance
                    for epoch in range(params['num_epochs']):
                        current_mse_weight = calculate_mse_weight(epoch, params['num_epochs'], initial_mse_weight,
                                                                  final_mse_weight, params['mse_annealing'])

                        start_time = time.time()
                        train_loss = train(model,
                                           train_dataloader,
                                           optimizer,
                                           device,
                                           RANGE_LIMIT,
                                           evaluation_bucket_nums[0],
                                           CLIP,
                                           criterion,
                                           scheduler,
                                           uniform_buckets=params['uniform_buckets'],
                                           mse_weight=current_mse_weight)

                        # evaluate model with selected metrics
                        training_metric_results, validation_metric_results = evaluation_manager(model,
                                                                                                (train_dataloader,
                                                                                                 val_dataloader),
                                                                                                optimizer,
                                                                                                criterion,
                                                                                                device,
                                                                                                metrics,
                                                                                                range_limit=RANGE_LIMIT,
                                                                                                evaluation_bucket_nums=evaluation_bucket_nums,
                                                                                                uniform_buckets=params[
                                                                                                    'uniform_buckets'],
                                                                                                mse_weight=current_mse_weight,
                                                                                                decay_rate=decay_rate,
                                                                                                dist_threshold=acc_dist_threshold)

                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                        # print metrics to console
                        print_metrics(epoch + 1, training_metric_results, validation_metric_results,
                                      epoch_mins=epoch_mins, epoch_secs=epoch_secs)

                        valid_acc = validation_metric_results['cumulative_ac'][0]
                        train_acc = training_metric_results['cumulative_ac'][0]

                        if valid_acc > best_acc_and_model[0]:
                            print(f'New best model found, accuracy: {valid_acc}')
                            # save best state dict under generic name so that it can be easily overridden with better versions
                            # will be renamed to actual model name later in the process
                            torch.save(model.state_dict(), f'{args.save_path}/best_acc_model.pt')
                            best_acc_and_model = [valid_acc, train_acc, log_file, epoch + 1]

                        # write all training information to tensorboard
                        record_metrics(writer, metrics, evaluation_bucket_nums, training_metric_results,
                                       validation_metric_results, step=epoch + 1)

                    train_accs.append(best_acc_and_model[1])
                    valid_accs.append(best_acc_and_model[0])

                    if best_acc_and_model[0] > best_kfold_acc:
                        print("New best Kfolsd acc")
                        best_kfold_acc = best_acc_and_model[0]
                        print("Evaluating it on test set")
                        training_metric_results, test_metric_results = evaluation_manager(model,
                                                                                          (train_dataloader,
                                                                                           test_dataloader),
                                                                                            optimizer,
                                                                                            criterion,
                                                                                            device,
                                                                                            metrics,
                                                                                            range_limit=RANGE_LIMIT,
                                                                                            evaluation_bucket_nums=evaluation_bucket_nums,
                                                                                            uniform_buckets=params[
                                                                                                'uniform_buckets'],
                                                                                            mse_weight=initial_mse_weight,
                                                                                            decay_rate=decay_rate,
                                                                                            dist_threshold=acc_dist_threshold)
                        print("Test Acc : ", test_metric_results['cumulative_ac'][0] )


                    # create directories for saved training metrics unless they have already been created
                    try:
                        os.mkdir(f'./{args.save_path}/train_metrics')
                        os.mkdir(f'./{args.save_path}/val_metrics')
                    except:
                        pass

                    if args.save_pretrained_models:
                        try:
                            os.mkdir(f'./{args.pretrained_path}')
                        except:
                            pass
                        lr = params['lr']
                        epochs = params['num_epochs']
                        g_toks = params['goal_tokens']
                        coral = params['coral']
                        torch.save(model.state_dict(),
                                   f'{args.pretrained_path}/d-{dataset_name}_a-{arch_name}_lr-{lr}_e-{epochs}_g-{g_toks}_c-{coral}.pt')

                    # save the models final performance to file, so that it can be automatically parsed later
                    pickle.dump(training_metric_results, open(f'{args.save_path}/train_metrics/{log_file}', 'wb'))
                    pickle.dump(validation_metric_results, open(f'{args.save_path}/val_metrics/{log_file}', 'wb'))

                    # todo: write script to parse metric outputs and convert them to an excel spreadsheet
                    writer.close()
                    

        print(train_accs)
        print(valid_accs)
        print(f"10-fold Training Accuracy Avg -- {float(np.mean(np.array(train_accs)))}")
        print(f"10-fold Training Accuracy Std -- {float(np.std(np.array(train_accs)))}")
        print(f"10-fold Validation Accuracy Avg -- {float(np.mean(np.array(valid_accs)))}")
        print(f"10-fold Validation Accuracy Std -- {float(np.std(np.array(valid_accs)))}")
    print("Training Complete")

    """
    best_acc = best_acc_and_model[0]
    best_acc_model_name = best_acc_and_model[2]
    best_acc_model_epoch = best_acc_and_model[3]
    print(f"Best model (acc): {best_acc_model_name}, {best_acc}")
    # rename the best saved state dict with the corect model name for convenience
    os.rename(f'{args.save_path}/best_acc_model.pt',
              f'{args.save_path}/best_acc_model_{best_acc_model_name},saved_epoch-{best_acc_model_epoch}.pt')
    """
