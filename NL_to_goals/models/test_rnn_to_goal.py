from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

"""
Contains pytorch architectures for rnn to goal models. Note that documentation is currently incomplete
"""


class Encoder(nn.Module):
    """
    Takes a tokenize input sequence and maps it to a vector encoding. This vector can then be
    used as the input to a classification or regression module.
    """
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tensor:
        embedded = self.dropout(self.embedding(src))
        outputs, __ = self.rnn(embedded)
        
        return outputs

class Attention(nn.Module):
    """
    Pytorch module for applying attention to a vector encoding
    """
    def __init__(self,
                 enc_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim

        self.attn_in = enc_hid_dim * 2

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                encoder_outputs: Tensor) -> Tensor:

        encoder_outputs = encoder_outputs.permute(1, 0, 2) #because GRUs return in form (seq_length, batch, num_directions*hidden_size)
        energy = torch.tanh(self.attn(encoder_outputs))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)



class Regressor(nn.Module):
    """
    #Module for predicting the goal values for inputs given hidden state outputs from an encoder and attention weights
    """

    def __init__(self,
                 enc_hid_dim: int,
                 num_goals: int,
                 attention_modules:nn.ModuleList):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.num_goals = num_goals
        self.attention_modules = attention_modules

        self.regress_in = enc_hid_dim * 2 * num_goals

        self.regress = nn.Linear(self.regress_in, num_goals)

    def forward(self,
                encoder_outputs: Tensor) -> Tensor:

        #create a weighted attention vector for each goal
        weighted_encoder_outputs = []
        for i,attention_module in enumerate(self.attention_modules):            
            a = attention_module(encoder_outputs)
            a = a.unsqueeze(1)
            temp_encoder_outputs = encoder_outputs.permute(1, 0, 2)
            weighted_encoder_rep = torch.bmm(a, temp_encoder_outputs)
            weighted_encoder_outputs.append(weighted_encoder_rep)

        #concatenate the weighted encoder outputs into a single tensor
        weighted_encoder_output = torch.cat(weighted_encoder_outputs, dim=2)

        #run the concatenated vector through a linear layer to provide predictions
        regression_output = self.regress(weighted_encoder_output).squeeze()
        return regression_output

class Classifier(nn.Module):
# class Bucketer(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 num_goals: int,
                 num_goal_buckets: int,
                 attention_modules:nn.ModuleList):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.num_goals = num_goals
        self.attention_modules = attention_modules

        self.lin_in = enc_hid_dim * 2

        self.linear_networks = nn.ModuleList([nn.Linear(self.lin_in, num_goal_buckets) for module in attention_modules])

    def forward(self,
                encoder_outputs: Tensor) -> Tensor:

        #create a weighted attention vector for each goal
        per_goal_bucket_probs = []
        for i, goal_modules in enumerate(zip(self.attention_modules, self.linear_networks)):            
            attention_module = goal_modules[0]
            linear_module = goal_modules[1]
            
            a = attention_module(encoder_outputs)
            a = a.unsqueeze(1)
            temp_encoder_outputs = encoder_outputs.permute(1, 0, 2)
            weighted_encoder_rep = torch.bmm(a, temp_encoder_outputs)
            energies = linear_module(weighted_encoder_rep) #leave dim 1 to represent the goal
            bucket_probs = F.log_softmax(energies, dim = 2) #should be batch x num_buckets
            per_goal_bucket_probs.append(bucket_probs) 

        goal_bucket_probs = torch.cat(per_goal_bucket_probs, dim=1)
        return goal_bucket_probs

class Seq2GoalValue(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 regressor: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.regressor = regressor

    def forward(self,
                src: Tensor) -> Tensor:

        batch_size = src.shape[1]

        encoder_outputs = self.encoder(src)

        outputs = self.regressor(encoder_outputs)

        return outputs


class Seq2GoalClass(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self,
                src: Tensor, selections) -> Tensor:

        encoder_outputs = self.encoder(src)

        outputs = self.classifier(encoder_outputs)

        return outputs