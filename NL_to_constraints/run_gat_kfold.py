import torch
import argparse
import random
# sys.path.append('../NL_constraints_data_prep')
from NL_constraints_data_prep.Utils.Dataset import setup_datasets, prepare_data
import itertools
import numpy as np
from run_gat import call_trainer



from utils.Trainer import Trainer
from utils.constants import *


if __name__ == "__main__":
    random.seed(1)
    parser = argparse.ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--warmup_steps", default=200, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--restart_from_conf", default=0, type=int, help="configuration to restart from")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--use_saved_model", action="store_true",
                        help="Use model pretrained on synthetic augmented data")
    parser.add_argument("-d", "--datasets", nargs="+", help="The names of the datasets to use", required=True)
    args = parser.parse_args()

    for data in args.datasets:
        assert data in ['human_augmented', 'v1', 'v2', 'v3',
                        'syn', 'aug'], "One of the arguments you provided to the datasets variable is invalid. The set of possible options are ['human_augmented', 'v1', 'v2', 'v3', 'syn', 'aug']"
    if 'aug' in args.datasets:
        assert 'syn' in args.datasets, "Aug can only be used in association with syn, to load the synthetic augmented dataset"
    best_valid_acc = 0
    search_space = {
        "lr": [3.5e-5],
        "batch_size": [32],
        "weight_decay": [0.001],
        "enc_embedding": [40],
        "num_train_epochs": [15],
        "optimizer": ['Adam'],
        "scheduler": ['Cosine'],
        "dropout": [0.5],
        "hidden_size": [64],
        "Glove": [False],
        "num_heads": [2],
        "model_type": ['roberta-con-tokens'],
        "pretrained_model": [None],
        "selections": ['bert_no_map'],
        # "axe_delta": [0.001],
        "oaxe_c": [1.1],
        "lambda": [0.1]
    }
    for num, conf in enumerate(
            itertools.product(search_space["lr"], search_space["batch_size"], search_space["weight_decay"],
                              search_space["num_train_epochs"], search_space["optimizer"],
                              search_space["enc_embedding"], search_space["dropout"],
                              search_space["hidden_size"], search_space["Glove"], search_space["num_heads"],
                              search_space["scheduler"], search_space['model_type'], search_space['oaxe_c'],
                              search_space['lambda'],
                              search_space['pretrained_model'], search_space['selections'])):
        if num < args.restart_from_conf:
            continue
        train_accs = []
        valid_accs = []
        best_valid_acc = float("-inf")
        for fold in range(10):
            print(f"Starting Fold: {fold}")
            prepare_data(args.datasets, kfold=True, fold = fold)
            print(best_valid_acc)
            best_valid_acc, train_accs, valid_accs = call_trainer(args, num, conf, kfold=True, fold=fold, best_valid_acc=best_valid_acc, train_accs=train_accs, valid_accs=valid_accs)

            # assert best_model is not None, "the trainer should return the model with the highest validation accuracy " \
            #                                "after training "
            # if 'roberta' in conf[11]:
            #     model_name = 'roberta-base'
            #     bert_model = AutoModel.from_pretrained(model_name)
            #     bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            #     bert_config = AutoConfig.from_pretrained(model_name)
            #     train_dataset, valid_dataset, test_dataset = setup_datasets(args.datasets, include_unk=True,
            #                                                                      tokenizer= bert_tokenizer, kfold = True)
            #     if 'con-tokens' in conf[11]:
            #         # Add constraint tokens to the dictionary so that they are tokenized
            #         # as individual tokens rather than subtokens
            #         bert_tokenizer.add_tokens(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
            #         bert_model.resize_token_embeddings(len(bert_tokenizer))
            # else:
            #     train_dataset, valid_dataset, test_dataset = setup_datasets(args.datasets, include_unk=True, kfold = True)
            #     bert_model = None
            #     bert_tokenizer = None
            #     bert_config = None
            # src_vocab = train_dataset.get_vocab()
            # # print(len(src_vocab.get_stoi()))
            # # sdfsdf
            # print("Training config: ", num)
            # if 'v1' in args.datasets or 'v2' in args.datasets or 'v3' in args.datasets:
            #     dataset = 'human'
            # elif 'syn' in args.datasets:
            #     if 'aug' in args.datasets:
            #         dataset = 'synthetic_augmented'
            #     else:
            #         dataset = 'synthetic'
            # elif 'ha' in args.datasets:
            #     dataset = 'human_augmented'
            #
            #
            # params = {
            #     'input_dim': len(src_vocab),
            #     'constraint_dim': len(CONSTRAINT_TYPES),
            #     'value_dim': len(VALUE_TYPES),
            #     'batch_size': conf[1],
            #     'enc_emb_dim': conf[5],
            #     'enc_hidden_dim': conf[7],
            #     'dec_hidden_dim': conf[7],
            #     'feedforward_dim': 64,
            #     'num_heads': conf[9],
            #     'dropout': conf[6],
            #     'clip': 1,
            #     'optimizer': conf[4],
            #     'scheduler': conf[10],
            #     'lr': conf[0],
            #     # 'min_lr': 0.0001,
            #     'weight_decay': conf[2],
            #     'num_epochs': conf[3],
            #     'glove': conf[8],
            #     'loss': 'OaXE',
            #     'model_type': conf[11],
            #     'oaxe_c': conf[12],
            #     'lambda': conf[13],
            #     'dataset': dataset,
            #     'use_saved_model': args.use_saved_model,
            #     'selections': conf[15],
            #     'pretrained_model': conf[14],
            #     'fold': fold,
            #     'iskfold': True
            # }
            #
            # conf_path = f'Outputs/{params["dataset"]}/{params["model_type"]}/'
            # c = 'conf_' + str(num)
            # print(params)
            #
            # assert params['selections'] in ['bert_no_map',
            #                                 None], "selections parameter needs to be one of [None, 'bert_no_map']"
            # assert params['pretrained_model'] in ['synthetic', 'synthetic_augmented',
            #                                       None], "pretrained model parameter needs to be one of [None, 'synthetic', 'synthetic_augmented']"
            #
            # trainer = Trainer(args,
            #                   params,
            #                   train_dataset,
            #                   valid_dataset,
            #                   test_dataset,
            #                   src_vocab,
            #                   bert_tokenizer,
            #                   bert_model,
            #                   bert_config,
            #                   kfold = True
            #                   )
            #
            # train_acc, valid_acc = trainer.train(num, best_valid_acc)
            # train_accs.append(train_acc)
            # valid_accs.append(valid_acc)
            #
            # if valid_acc > best_valid_acc:
            #     best_valid_acc = valid_acc
            #     best_model = deepcopy(trainer.model)
            #     trainer.save_model(params['num_epochs'], 'KFold', conf_path + c + '/model_kfold_best.pt')
            # torch.cuda.empty_cache()
            # del trainer.model
        print(train_accs)
        print(valid_accs)
        print(f"10-fold Training Accuracy Avg -- {float(np.mean(np.array(train_accs)))}")
        print(f"10-fold Training Accuracy Std -- {float(np.std(np.array(train_accs)))}")
        print(f"10-fold Validation Accuracy Avg -- {float(np.mean(np.array(valid_accs)))}")
        print(f"10-fold Validation Accuracy Std -- {float(np.std(np.array(valid_accs)))}")

