import csv

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import deepcopy
from os.path import expanduser
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from NL_to_constraints.models.rnn_GAT import Encoder, Decoder, Seq2Con, simpleDecoder, simpleNetworkwAttn, BertToConstraintClass
from NL_constraints_data_prep.Utils.data_utils_helpers import decode_batch, decode_constraint
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.tensorboard import SummaryWriter
from .constants import *
from .loss_functions import *
from .training_utils import *
from functools import partial


class Trainer:
    def __init__(
            self,
            args=None,
            training_parameters=None,
            train_dataset=None,
            valid_dataset=None,
            test_dataset=None,
            src_vocab=None,
            tokenizer = None,
            huggingface_model = None,
            huggingface_config = None,
            kfold = False
    ):
        self.args = args
        self.params = training_parameters
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.src_vocab = src_vocab
        self.tokenizer = tokenizer
        self.kfold = kfold
        self.huggingface_model = huggingface_model
        self.huggingface_config = huggingface_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_saved_model = args.use_saved_model
        if self.use_saved_model:
            self.pretraining_prefix = 'pretrained_'
        else:
            self.pretraining_prefix = ''

        # initialize data loaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.params['batch_size'],
                                           collate_fn=self.collate_fn,
                                           shuffle=True)
        self.val_dataloader = DataLoader(self.valid_dataset, batch_size=self.params['batch_size'],
                                         collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.params['batch_size'],
                                          collate_fn=self.collate_fn)
        # weights = [0.0005] + [0.9995 / (le    n(CONSTRAINT_TYPES) * len(VALUE_TYPES) - 1) for i in
        #                       range(len(CONSTRAINT_TYPES) * len(VALUE_TYPES) - 1)]
        # weights = torch.Tensor(weights).to(self.device)
        # weights.requires_grad = False
        self.criterion = nn.CrossEntropyLoss()

        self.con_tokens = False
        if 'con-tokens' in self.params['model_type']:
            self.con_tokens = True

        self.setup_model()

        self.t_total = len(self.train_dataloader) * self.params['num_epochs']

        if self.params['optimizer'] == 'Adam':
            if 'roberta' in self.params['model_type']:
                # If using a huggingface model, set different learning rates for encoder/decoder
                bert_params = []
                decoder_params = []
                for name, param in self.model.named_parameters():
                    if 'bert_model' in name:
                        bert_params.append(param)
                    else:
                        decoder_params.append(param)
                self.optimizer = optim.AdamW(
                    [{"params": bert_params, "lr":1e-5, "weight_decay":self.params['weight_decay']},
                     {"params": decoder_params, "lr":self.params['lr'], "weight_decay":self.params['weight_decay']}
                     ]
                )
            else:
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay']
                )
        elif self.params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay']
            )
        if self.params['scheduler'] == 'StepLR':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
            )
        elif self.params['scheduler'] == 'Cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
            )

    def train(self, conf_num, best_kfold_acc = 0):
        '''
        Outer train function which trains and evaluates the model for the given number of epochs.
        Logs/Metrics written to tensorboard
        Outputs of the best performing model are saved in a log file
        :param conf_num: configuration number in case you are using a grid search
        :return: Null
        '''
        conf_path = f'Outputs/{self.params["dataset"]}/{self.params["model_type"]}/'
        c = 'conf_' + str(conf_num)

        self.init_logging_tensorboard(conf_num, conf_path, c)

        best_valid_acc = float("-inf")
        best_train_acc = float("-inf")
        training_accs = []
        for epoch in range(self.params['num_epochs']):
            start_time = time.time()

            self.compute_temperature(epoch)
            train_loss, train_acc = self.train_loop(self.train_dataloader, loss_func=self.params['loss'])
            valid_loss, valid_acc = self.evaluate(self.model, self.val_dataloader, loss_func=self.params['loss'])
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            training_accs.append(train_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = deepcopy(self.model)
                self.save_model(epoch, train_loss, conf_path + c + '/model.pt')
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_model = deepcopy(self.model)
                self.save_model(epoch, train_loss, conf_path + c + '/best_train_model.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}, Train Acc: {train_acc: .3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}, Val Acc: {valid_acc: .3f}')
            print(f'\t Learning Rate: {self.scheduler.get_last_lr()}')
            self.writer.add_scalar('Loss/train', train_loss, epoch * len(self.train_dataloader))
            self.writer.add_scalar('Loss/valid', valid_loss, epoch * len(self.train_dataloader))
            self.writer.add_scalar('Accuracy/train', train_acc, epoch * len(self.train_dataloader))
            self.writer.add_scalar('Accuracy/valid', valid_acc, epoch * len(self.train_dataloader))
            if check_early_stop(training_accs):
                print("---------------Early Stop--------------")
                break
        self.logger.info('Best Validation Accuracy:')
        self.logger.info(best_valid_acc)
        self.train_logger.info('Training loss for model with best training accuracy: ')
        self.train_logger.info(best_train_acc)
        _ = self.evaluate(best_model, self.val_dataloader, 'eval', loss_func=self.params['loss'], logger=self.logger,
                          best=True)
        if not self.kfold:
            self.test_file = init_output_file(conf_path, c)
            _ = self.evaluate(best_model, self.test_dataloader, 'test', loss_func=self.params['loss'], logger=self.logger,
                          best=True)
        else:
            if best_valid_acc > best_kfold_acc:
                self.test_file = init_output_file(conf_path, c)
                print("Evaluating best model so far on testing set")
                _ = self.evaluate(best_model, self.test_dataloader, 'test', loss_func=self.params['loss'],
                                  logger=self.logger,
                                  best=True)
        _ = self.evaluate(best_train_model, self.train_dataloader, 'train', loss_func=self.params['loss'],
                          logger=self.train_logger, best=True)
        end_time = time.time()
        print("-----------------------------------------------------------------------------------------------")
        return best_train_acc, best_valid_acc

    def train_loop(self, iterator, loss_func='CE'):
        '''
        Code for running one epoch of the training process
        :param iterator: dataloader
        :return: train_loss: float
        '''
        self.model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0].to(self.device)
            constraints = batch[1].to(self.device)
            goals = batch[2].to(self.device)
            # maps = batch[3].to(self.device)
            # selections = batch[4].to(self.device)
            self.optimizer.zero_grad()
            output_logits, output_c, output_v = self.model(src, constraints)

            if self.params['loss'] == 'AXE':
               assert tuple(output_logits.shape) == (self.params['batch_size'], 8, len(CONSTRAINT_TYPES) * len(VALUE_TYPES) + 1), "Your output logits are not in the correct shape. Correct shape is [Batch_size x num_constraints x output_dim]"
            else:
                assert output_logits.shape[1] == 8 and output_logits.shape[2] == (len(CONSTRAINT_TYPES) * len(
                    VALUE_TYPES)), "Your output logits are not in the correct shape. Correct shape is [Batch_size x num_constraints x output_dim]"

            loss, b_correct, b_total, predictions = self.compute_loss_and_accuracy(constraints, output_logits, loss_func)
            loss.backward()



            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])

            self.optimizer.step()
            self.scheduler.step()
            total += b_total
            correct += b_correct

            epoch_loss += loss.item()
            del loss
            del output_logits
        return epoch_loss / len(iterator), float(correct) / total

    def compute_loss_and_accuracy(self, constraints, output_logits, loss_func='CE'):
        '''
        Function to compute loss function based on whether you are applying Cross entropy or Aligned Cross Entropy
        This function also computes accuracy of a batch and returns predictions based on the logits for this batch.
        Parameters:
        ---------
        constraints: ``torch.LongTensor``, required
        A tensor which contains the target labels
        output_logits: ``torch.FloatTensor`` required
        A tensor which contains the output logits from the model
        loss_function: ``str``, ['CE', 'AXE']
        '''
        con_output_dim = self.params['constraint_dim']
        val_output_dim = self.params['value_dim']
        con = constraints.view(-1)
        if loss_func == 'CE':
            output_dim = con_output_dim * val_output_dim
            output_logits = output_logits.view(-1, output_dim)
            predictions = make_predictions(output_logits)
            inv_loss, predictions = invalid_constraint_loss(predictions)

            l = self.criterion(output_logits, con)
            loss = l + inv_loss
        elif loss_func == 'AXE':
            logit_lengths = torch.full((constraints.shape[0],), 8)
            target_lengths = torch.full((constraints.shape[0],), 8)
            l = axe_loss(output_logits, logit_lengths, constraints, target_lengths, BLANK_INDEX, self.params['axe_delta'])

            # output_dim = con_output_dim * val_output_dim + 1
            output_logits = output_logits.view(-1, output_logits.shape[2])
            predictions = make_predictions(output_logits)
            inv_loss, predictions = invalid_constraint_loss(predictions)
            loss = l + inv_loss
        elif loss_func == 'OaXE':

            oaxe_loss = oaxe(output_logits, constraints, self.criterion)

            output_logits = output_logits.view(-1, output_logits.shape[-1])
            predictions = make_predictions(output_logits)
            inv_loss, predictions = invalid_constraint_loss(predictions)
            ce_loss = self.criterion(output_logits, con)

            loss = (ce_loss * self.temperature + (1 - self.temperature) * oaxe_loss) + inv_loss

        correct, total = compute_accuracy(predictions, constraints)
        return loss, correct, total, predictions

    def plot_constraints(self):
        constraint_count = defaultdict(int)
        num_constraints_list = []
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            constraints = batch[1].to(self.device)
            temp_con = constraints.view(-1)
            non_empty = []
            for j in range(len(temp_con)):
                act_c, act_v = decode_constraint(temp_con[j].item())
                if not act_c == 0:
                    non_empty.append(act_c)
                constraint_count[INV_CON[act_c]] += 1
            num_constraints_list.append(len(non_empty) / float(constraints.shape[0]))
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            constraints = batch[1].to(self.device)
            temp_con = constraints.view(-1)
            non_empty = []
            for j in range(len(temp_con)):
                act_c, act_v = decode_constraint(temp_con[j].item())
                if not act_c == 0:
                    non_empty.append(act_c)
                constraint_count[INV_CON[act_c]] += 1
            num_constraints_list.append(len(non_empty) / float(constraints.shape[0]))
        print(num_constraints_list)
        print(f"Mean number of constraints - {(np.mean(num_constraints_list))}")
        del constraint_count['<NA>']
        # print(constraint_count)
        plt.bar(constraint_count.keys(), constraint_count.values(), color='b')
        plt.xticks(rotation = 45, ha = 'right')
        plt.xlabel('Constraint Class')
        plt.ylabel('Number of Examples in Dataset')
        plt.title('Distribution of Constraints in the Dataset')
        plt.tight_layout()
        plt.savefig('Constraints_distribution.png')



    def compute_temperature(self, epoch):
        '''
        Compute temperature based on the current epoch
        T = max(0, 1 - c ^ (epoch - lambda * N)
        '''
        self.temperature = max(0, 1 - self.params['oaxe_c'] ** (epoch - self.params['lambda'] * self.params['num_epochs']))


    def evaluate(self, model, iterator, mode=None, loss_func='CE', logger=None, best=False):
        '''
        Eval loop
        :param model: model to evaluate on
        :param iterator: dataloader
        :param logger: logger to use for saving outputs
        :param best: whether or not the best performing model is being tested
        :param mode: Whether you are running this function on the validation set or training set ['train', 'eval', 'test', None]
        :return: eval_loss: float
        '''
        assert mode in ['train', 'eval', 'test', None], "evaluate mode must be either train, eval, test or None"


        model.eval()
        con_output_dim = self.params['constraint_dim']
        val_output_dim = self.params['value_dim']
        epoch_loss = 0
        num_total = 0
        num_correct = 0


        for i, batch in enumerate(iterator):
            src = batch[0].to(self.device)
            constraints = batch[1].to(self.device)
            goals = batch[2].to(self.device)
            maps = batch[3].to(self.device)
            selections = batch[4].to(self.device)

            self.optimizer.zero_grad()
            output_logits, output_c, output_v = model(src, constraints)
            loss, b_correct, b_total, predictions = self.compute_loss_and_accuracy(constraints, output_logits, loss_func)

            text = self.convert_to_text(src)

            if best == True:
                # predictions = output_logits.topk(k=1, dim=1)[1]
                temp_con = constraints.view(-1)
                temp_maps = maps.view(-1)
                prediction_row = []
                target_row = []
                if mode == 'train':
                    file = self.csv_file
                elif mode == 'eval':
                    file = self.valid_file
                elif mode == 'test':
                    file = self.test_file
                for j in range(len(predictions)):
                    # Save predictions
                    pred_c, pred_v = decode_constraint(predictions[j].item())
                    act_c, act_v = decode_constraint(temp_con[j].item())
                    prediction_row.append(INV_CON[pred_c] + '- [' + INV_VAL[pred_v] + ']')
                    target_row.append(INV_CON[act_c] + '- [' + INV_VAL[act_v] + ']')
                    if j%8 == 7:
                        with open(file, 'a') as myfile:
                            csv_writer = csv.writer(myfile)
                            csv_writer.writerow([str(temp_maps[int(j/8)].item())] + ['Actual#' + str(int(j/8))] + [text[int(j/8)].replace("<pad>", "")] + target_row)
                            csv_writer.writerow([str(temp_maps[int(j/8)].item())] + ['Prediction#' + str(int(j/8))] + ['--'] + prediction_row)
                        target_row = []
                        prediction_row = []
                        logger.info('----SEQUENCE COMPLETE----')

                with open(file, 'a') as myfile:
                    csv_writer = csv.writer(myfile)
                    csv_writer.writerow(['-', '-', '-', '-', '-', '-', 'BATCH', 'COMPLETE', '-', '-', '-', '-', '-', '-', '-'])
                logger.info('----BATCH COMPLETE----')

            con = constraints.view(-1)

            epoch_loss += loss.item()
            num_total += b_total
            num_correct += b_correct
            del src
            del loss
        return epoch_loss / len(iterator), float(num_correct) / num_total

    def convert_to_text(self, batch):
        '''
        Convert a batch of outputs to text
        Args:
            batch:

        Returns: text = list[str] -> List of decoded strings
        '''
        if 'roberta' in self.params['model_type']:
            text = self.tokenizer.batch_decode(batch['input_ids'])
        else:
            text = decode_batch(batch, self.src_vocab)
        return text

    def save_model(self, epoch, loss, path):
        '''
        Save model, optimizer and scheduler to the state dict for checkpointing
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def setup_model(self):
        '''
        Setup goals to constraints model.
        Load pretrained models to initialize model if pretraining is selected
        :return: Null
        '''

        src_matrix_len = len(self.src_vocab.get_stoi())
        src_weights_matrix = None
        if self.params['glove']:
            f1 = open(expanduser("~") + '/Glove/glove_dict_100.pkl', 'rb')
            glove_dict = pickle.load(f1)

            src_weights_matrix = np.zeros((src_matrix_len, self.params['enc_emb_dim']))
            words_found = 0

            for i, word in enumerate(self.train_dataset.get_vocab().get_itos()):
                try:
                    src_weights_matrix[i] = glove_dict[word]
                    words_found += 1
                except KeyError:
                    src_weights_matrix[i] = np.random.normal(scale=0.6, size=(self.params['enc_emb_dim'],))

        enc = Encoder(len(self.src_vocab.get_stoi()), self.params['enc_emb_dim'], self.params['enc_hidden_dim'],
                      self.params['dec_hidden_dim'],
                      self.params['dropout'], emb_weight_matrix=src_weights_matrix, bidirectional=True)
        dec_con = Decoder(len(CONSTRAINT_TYPES), len(VALUE_TYPES),
                          self.params['dec_hidden_dim'],
                          self.params['feedforward_dim'], self.params['num_heads'], self.params['dropout'])



        if self.params['model_type'] == 'simple':
            simple_dec = simpleDecoder(enc.enc_hid_dim * 2, self.device, loss=self.params['loss'])
            # self.model = simpleNetwork(enc, self.device, loss = self.params['loss']).to(self.device)
            self.model = Seq2Con(enc, simple_dec, self.device, enc_dec_attn=False).to(self.device)
        elif self.params['model_type'] == 'simplewAttn':
            self.model = simpleNetworkwAttn(enc, self.device, loss = self.params['loss']).to(self.device)
        elif self.params['model_type'] == 'gat':
            self.model = Seq2Con(enc, dec_con, self.device, enc_dec_attn=True).to(self.device)
        elif self.params['model_type'] == 'gat_no_enc_dec_attn':
            self.model = Seq2Con(enc, dec_con, self.device, enc_dec_attn=False).to(self.device)
        elif self.params['model_type'] == 'roberta-base' or self.params['model_type'] == 'roberta-con-tokens':

            simple_dec = simpleDecoder(self.huggingface_config.hidden_size, self.device, loss=self.params['loss'])
            self.model = BertToConstraintClass(self.huggingface_model, simple_dec, self.huggingface_config.hidden_size, con_tokens = self.con_tokens).to(self.device)
        elif self.params['model_type'] == 'roberta-gat':

            dec_con = Decoder(len(CONSTRAINT_TYPES), len(VALUE_TYPES),
                              self.params['dec_hidden_dim'],
                              self.params['feedforward_dim'], self.params['num_heads'], self.params['dropout'])
            self.model = BertToConstraintClass(self.huggingface_model, dec_con, self.huggingface_config.hidden_size,
                                               loss=self.params['loss']).to(self.device)

        if 'roberta' not in self.params['model_type']:
            self.model.apply(init_weights)
        if self.use_saved_model:
            model_dict = torch.load(f'saved_models/best_model_{self.params["model_type"]}_{self.params["pretrained_model"]}.pt')['model_state_dict']
            self.model.load_state_dict(model_dict)
            print(f"Loading saved model from saved_models/best_model_{self.params['model_type']}_{self.params['pretrained_model']}.pt")
    def init_logging_tensorboard(self, conf_num, conf_path, conf_name):
        '''
        Initialize logging and tensorboard details
        Create csvs required for storing output data
        :return: Null
        '''
        if self.params['iskfold']:
            log_file = f"run_{conf_num}_lr:{self.params['lr']},batch_size:{self.params['batch_size']},weight_decay:{self.params['weight_decay']},epochs:{self.params['num_epochs']}, optimizer:{self.params['optimizer']}, scheduler: {self.params['scheduler']}, dropout:{self.params['dropout']}, hidden_size:{self.params['enc_hidden_dim']}, model: {self.params['model_type']}, c: {self.params['oaxe_c']}, lambda:{self.params['lambda']}, pretrained_model:{self.params['pretrained_model']}, sel:{self.params['selections']}, fold:{self.params['fold']}|"
            self.writer = SummaryWriter(
                f'standard_model/Kfold/{self.params["dataset"]}/{self.pretraining_prefix}{self.params["model_type"]}/' + log_file)
        else:
            log_file = f"run_{conf_num}_lr:{self.params['lr']},batch_size:{self.params['batch_size']},weight_decay:{self.params['weight_decay']},epochs:{self.params['num_epochs']}, optimizer:{self.params['optimizer']}, scheduler: {self.params['scheduler']}, enc_embedding:{self.params['enc_emb_dim']}, dropout:{self.params['dropout']}, hidden_size:{self.params['enc_hidden_dim']}, model: {self.params['model_type']}, c: {self.params['oaxe_c']}, lambda:{self.params['lambda']}, pretrained_model:{self.params['pretrained_model']}, sel:{self.params['selections']}|"
            self.writer = SummaryWriter(f'standard_model/User_Study_Split/{self.params["dataset"]}/{self.pretraining_prefix}{self.params["model_type"]}/' + log_file)

        if not os.path.exists(conf_path + '/' + conf_name):
            os.makedirs(conf_path + '/' + conf_name)
        file = conf_path + '/' + conf_name + '/' + 'config.txt'
        self.csv_file = conf_path + '/' + conf_name + '/' + 'train_outputs.csv'
        self.valid_file = conf_path + '/' + conf_name + '/' + 'valid_outputs.csv'
        # self.test_file = conf_path + '/' + conf_name + '/' + 'test_outputs.csv'
        labels = ['', 'Map', 'Input Text', 'constraint_1', 'constraint_2', 'constraint_3', 'constraint_4', 'constraint_5', 'constraint_6', 'constraint_7', 'constraint_8']

        with open(self.csv_file, 'w') as myfile:
            csv_writer = csv.writer(myfile)
            csv_writer.writerow(labels)
        with open(self.valid_file, 'w') as myfile:
            csv_writer = csv.writer(myfile)
            csv_writer.writerow(labels)
        # with open(self.test_file, 'w') as myfile:
        #     csv_writer = csv.writer(myfile)
        #     csv_writer.writerow(labels)

        F = open(file, 'w+')
        F.write("{\n")
        for i, key in enumerate(self.params.keys()):
            F.write('\'' + key + '\'')
            F.write(": ")
            F.write(str(self.params[key]))
            if i < len(self.params) - 1:
                F.write(",")
            F.write("\n")
        F.write("}")
        F.close()

        self.logger = logging.getLogger(conf_name)
        hdlr = logging.FileHandler(conf_path + conf_name + '/' + conf_name + '.log', mode='w')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.train_logger = logging.getLogger(conf_name + '_train')
        hdlr = logging.FileHandler(conf_path + conf_name + '/' + conf_name + '_train' + '.log', mode='w')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.train_logger.addHandler(hdlr)
        self.train_logger.setLevel(logging.INFO)
        self.train_logger.propagate = False

    def collate_fn(self, batch):
        '''
            convert outputs of the iterator into a batch of tensors that can be used as an input to the model.
            :param batch: (selection, constraint, goal, map, txt)
            :return: (selection, constraint, goal, map, txt) -> Torch.tensor
        '''
        texts, constraints, selections, goals, maps = [], [], [], [], []
        for selection, constraint, goal, map, txt in batch:
            if self.params['selections'] == 'bert_no_map':
                selection_text = '[SEL]:'
                for k in range(len(selection)):
                    if '<eos>' not in selection[k] and '<pad>' not in selection[k] and '<sos>' not in selection[k]:
                        selection_text += selection[k][0] + ' = ' + str(selection[k][1]) + '|'
                selection_text = selection_text[:-1]
                selection_text += '[/SEL]'
                txt += ' ' + selection_text
            if self.con_tokens:
                txt = 'C1C2C3C4C5C6C7C8' + txt
            texts.append(txt)
            selection_var = deepcopy(selection)
            for s in selection_var:
                if type(s[0]) == str:
                    s[0] = COUNTRY_MAP[s[0]]
                    if s[0] < 3:
                        s[1] = COUNTRY_MAP[s[1]]
            selections.append(torch.tensor(selection_var))
            constraint = torch.tensor(constraint)
            constraints.append(constraint)
            goals.append(torch.tensor(goal))
            maps.append(map)
        if 'roberta' in self.params['model_type']:
            texts = self.tokenizer(texts, padding=True)
            for key in texts.keys():
                texts[key] = torch.LongTensor(texts[key])
        else:
            texts = pad_sequence(texts, batch_first=False, padding_value=self.src_vocab.get_stoi()['<pad>'])
        goals = torch.stack(goals)
        maps = torch.tensor(maps)
        constraints = torch.stack(constraints)
        selections = torch.stack(selections)
        return texts, constraints, goals, maps, selections

    def collate_fn_rnn(self, batch):
        '''
        DEPRECATED
        convert outputs of the iterator into a batch of tensors that can be used as an input to the model.
        :param batch: (selection, constraint, goal, map, txt)
        :return: (selection, constraint, goal, map, txt) -> Torch.tensor
        '''
        texts, constraints, selections, goals, maps = [], [], [], [], []
        for selection, constraint, goal, map, txt in batch:
            texts.append(txt)
            for s in selection:
                if type(s[0]) == str:
                    s[0] = COUNTRY_MAP[s[0]]
                    if s[0] < 3:
                        s[1] = COUNTRY_MAP[s[1]]
            selections.append(torch.tensor(selection))
            constraint = torch.tensor(constraint)
            constraints.append(constraint)
            goals.append(torch.tensor(goal))
            maps.append(map)
        texts = pad_sequence(texts, batch_first=False, padding_value=self.src_vocab.get_stoi()['<pad>'])
        goals = torch.stack(goals)
        maps = torch.tensor(maps)
        constraints = torch.stack(constraints)
        selections = torch.stack(selections)
        return texts, constraints, goals, maps, selections

    def collate_fn_transformer(self, batch):
        '''
        DEPRECATED
        convert outputs of the iterator into a batch of tensors that can be used as an input to the model.
        :param batch: (selection, constraint, goal, map, txt)
        :return: (selection, constraint, goal, map, txt) -> Torch.tensor
        '''
        texts, constraints, selections, goals, maps = [], [], [], [], []
        texts_input_ids, texts_token_ids, texts_attention = [], [], []
        for selection, constraint, goal, map, txt in batch:
            texts.append(txt)
            for s in selection:
                if type(s[0]) == str:
                    s[0] = COUNTRY_MAP[s[0]]
                    if s[0] < 3:
                        s[1] = COUNTRY_MAP[s[1]]
            selections.append(torch.tensor(selection))
            constraint = torch.tensor(constraint)
            constraints.append(constraint)
            goals.append(torch.tensor(goal))
            maps.append(map)

        texts = self.tokenizer(texts, padding=True)
        for key in texts.keys():
            texts[key] = torch.LongTensor(texts[key])
        goals = torch.stack(goals)
        maps = torch.tensor(maps)
        constraints = torch.stack(constraints)
        selections = torch.stack(selections)
        return texts, constraints, goals, maps, selections





def calc_entropy(input_tensor):

    entropy2 = Categorical(probs=p_tensor).entropy()
