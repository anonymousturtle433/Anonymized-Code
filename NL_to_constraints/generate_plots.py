import torch
import argparse
import sys
# sys.path.append('../NL_constraints_data_prep')
from NL_constraints_data_prep.Utils.Dataset import setup_datasets, prepare_data
import itertools
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel


from utils.Trainer import Trainer
from utils.constants import *

parser = argparse.ArgumentParser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
parser.add_argument("--warmup_steps", default=200, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--restart_from_conf", default=0, type=int, help="configuration to restart from")
parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--use_saved_model", action="store_true", help="Use model pretrained on synthetic augmented data")
parser.add_argument("-d", "--datasets", nargs="+", help="The names of the datasets to use", default="synthetic")
args = parser.parse_args()



if __name__ == "__main__":
    if args.overwrite_cache:
        prepare_data(args.datasets)

    search_space = {
        "lr": [5e-4],
        "batch_size": [8],
        "weight_decay": [0.001],
        "enc_embedding": [40],
        "num_train_epochs": [25],
        "optimizer": ['Adam'],
        "scheduler": ['Cosine'],
        "dropout": [0.5],
        "hidden_size": [64],
        "Glove": [False],
        "num_heads": [2],
        "model_type": ['roberta-base'],
        "pretrained_model": [None],
        "selections": [None],
        # "axe_delta": [0.001],
        "oaxe_c": [1.1],
        "lambda": [0.1]
    }
    for num, conf in enumerate(
            itertools.product(search_space["lr"], search_space["batch_size"], search_space["weight_decay"],
                              search_space["num_train_epochs"], search_space["optimizer"],
                              search_space["enc_embedding"], search_space["dropout"],
                              search_space["hidden_size"], search_space["Glove"], search_space["num_heads"],
                              search_space["scheduler"], search_space['model_type'], search_space['oaxe_c'], search_space['lambda'],
                              search_space['pretrained_model'], search_space['selections'])):
        if num < args.restart_from_conf:
            continue
        if 'roberta' in conf[11]:
            model_name = 'roberta-base'
            bert_model = AutoModel.from_pretrained(model_name)
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            bert_config = AutoConfig.from_pretrained(model_name)
            train_dataset, valid_dataset, test_dataset = setup_datasets(args.datasets, include_unk=True,
                                                                             tokenizer= bert_tokenizer)
            if 'con-tokens' in conf[11]:
                bert_tokenizer.add_tokens(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
                bert_model.resize_token_embeddings(len(bert_tokenizer))
        else:
            train_dataset, valid_dataset, test_dataset = setup_datasets(args.datasets, include_unk=True)
            bert_model = None
            bert_tokenizer = None
            bert_config = None
        src_vocab = train_dataset.get_vocab()
        # print(len(src_vocab.get_stoi()))
        # sdfsdf
        print("Training config: ", num)
        if 'v1' in args.datasets or 'v2' in args.datasets or 'v3' in args.datasets:
            dataset = 'human'
        elif 'syn' in args.datasets:
            if 'aug' in args.datasets:
                dataset = 'synthetic_augmented'
            else:
                dataset = 'synthetic'

        params = {
            'input_dim': len(src_vocab),
            'constraint_dim': len(CONSTRAINT_TYPES),
            'value_dim': len(VALUE_TYPES),
            'batch_size': conf[1],
            'enc_emb_dim': conf[5],
            'enc_hidden_dim': conf[7],
            'dec_hidden_dim': conf[7],
            'feedforward_dim': 64,
            'num_heads': conf[9],
            'dropout': conf[6],
            'clip': 1,
            'optimizer': conf[4],
            'scheduler': conf[10],
            'lr': conf[0],
            # 'min_lr': 0.0001,
            'weight_decay': conf[2],
            'num_epochs': conf[3],
            'glove': conf[8],
            'loss': 'OaXE',
            'model_type': conf[11],
            'oaxe_c': conf[12],
            'lambda': conf[13],
            'dataset': dataset,
            'use_saved_model': args.use_saved_model,
            'selections': conf[15],
            'pretrained_model': conf[14],
            'iskfold': False
        }
        print(params)

        trainer = Trainer(args,
                          params,
                          train_dataset,
                          valid_dataset,
                          test_dataset,
                          src_vocab,
                          bert_tokenizer,
                          bert_model,
                          bert_config
                          )

        trainer.plot_constraints()
        del trainer.model
        torch.cuda.empty_cache()
