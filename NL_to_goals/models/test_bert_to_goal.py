from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

"""Contains pytorch architectures for transformer/bert to goal models"""

class Classifier(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 num_goals: int,
                 num_goal_buckets: int,
                 dropout: float,
                 is_ordinal=False,
                 is_coral=False):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.num_goals = num_goals
        self.num_goal_buckets = num_goal_buckets
        self.is_ordinal = is_ordinal
        self.is_coral = is_coral
        if self.is_ordinal: #reduce the number of output buckets by 1, since making num_buckets - 1 predictions
            self.num_goal_buckets -= 1
        self.dropout = dropout



        self.pooling_layers = nn.ModuleList([nn.Linear(self.enc_hid_dim, self.enc_hid_dim) for i in range(num_goals)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for i in range(num_goals)])
        if self.is_coral: #if coral, only need 1 output from linear and a set of biases
            self.classifying_layers = nn.ModuleList([nn.Linear(self.enc_hid_dim, 1) for i in range(num_goals)])
            self.coral_biases = nn.Parameter(data=torch.zeros(1, self.num_goals, self.num_goal_buckets)) #batch dim, goal dim, buckets dim
        else:
            self.classifying_layers = nn.ModuleList([nn.Linear(self.enc_hid_dim, self.num_goal_buckets) for i in range(num_goals)])

    def forward(self,
                cls_hidden_states: Tensor, #batch x hidden dim or batch x num goals x hidden dim
                goal_tokens=False) -> Tensor: #batch x encoder hidden size
        """

        :param cls_hidden_states: Bert cls embedding
        :param goal_tokens: Boolean to indicate if we are using goal tokens
        :return: Probabilities of goal classes

        This function takes in the bert embeddings and predicts the probability of goals.
        It contains a classification layer for each goal with the number of buckets as the output
        """
        if not goal_tokens:
            cls_hidden_states = cls_hidden_states[:,None,:].expand((-1, self.num_goals, -1))

        per_goal_class_probs = []
        for i, goal_modules in enumerate(zip(self.pooling_layers, self.classifying_layers, self.dropout_layers)):
            goal_specific_hidden_state = cls_hidden_states[:,i,:]

            pooling_layer = goal_modules[0]
            classifying_layer = goal_modules[1]
            dropout_layer = goal_modules[2]

            pooled_output = torch.tanh(pooling_layer(goal_specific_hidden_state))
            pooled_output_dropout = dropout_layer(pooled_output)

            classification = classifying_layer(pooled_output_dropout)

            if self.is_coral:
                #apply coral biases
                classification = self.coral_biases[:,i,:] + classification #use coral biases for the particular goal
              
            if self.is_ordinal: #apply sigmoid to each bucket
                class_probs = torch.sigmoid(classification)  # should be batch x num_buckets
            else:
                class_probs = F.log_softmax(classification, dim=1)  # should be batch x num_buckets

            per_goal_class_probs.append(class_probs.unsqueeze(1))

        goal_class_probs = torch.cat(per_goal_class_probs, dim=1)

        return goal_class_probs

class Regressor(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 num_goals: int,
                 dropout: float):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.num_goals = num_goals
        self.dropout = dropout

        self.pooling_layers = nn.ModuleList([nn.Linear(self.enc_hid_dim, self.enc_hid_dim) for i in range(num_goals)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for i in range(num_goals)])
        self.classifying_layers = nn.ModuleList([nn.Linear(self.enc_hid_dim, 1) for i in range(num_goals)]) #since only predicting 1 number for each goal

    def forward(self,
                cls_hidden_state: Tensor,
                per_goal_attention=False) -> Tensor: #batch x encoder hidden size 
        """

        :param cls_hidden_state:  Bert cls embedding
        :param per_goal_attention: If true, assumes that input is tokenized with Goal1Tok Goal2Tok .... at the front
        :return: Predicts a goal value for each goal

        This function is used with the MSE loss where we are trying to predict a single goal value.
        """
        per_goal_prediction = []
        for i, goal_modules in enumerate(zip(self.pooling_layers, self.classifying_layers, self.dropout_layers)):            
            if per_goal_attention:
              goal_hidden_state = cls_hidden_state[:,i,:]
            else:
              goal_hidden_state = cls_hidden_state

            
            pooling_layer = goal_modules[0]
            classifying_layer = goal_modules[1]
            dropout_layer = goal_modules[2]

            pooled_output = torch.tanh(pooling_layer(goal_hidden_state))
            pooled_output_dropout = dropout_layer(pooled_output)
            classification = classifying_layer(pooled_output_dropout)
            per_goal_prediction.append(classification)

        goal_class_predictions = torch.cat(per_goal_prediction, dim=1)
        return goal_class_predictions

class SelectionsEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tensor:
        """

        :param src:  selections
        :return: encoded selections

        This function is used with one hot and partial one hot encoding of selections.

        """
        encoded = self.dropout(self.linear_relu_stack(src))
        return encoded

class BertToGoal(nn.Module):
    def __init__(self,
                 bert_model: nn.Module,
                 classifier: nn.Module):
        super().__init__()

        self.bert_model = bert_model
        self.classifier = classifier

    def forward(self,
                src) -> Tensor: #batch x encoder hidden size #assumes that src is a batch in the style of a transformers.tokenization_utils_base.BatchEncoding
        """
        This function is not used anymore
        Developed initially by Nathan
        """
        bert_output = self.bert_model(**src)
        cls_hidden_state = bert_output[0][:,1,:]
        goal_class_probs = self.classifier(cls_hidden_state)

        return goal_class_probs

class BertToGoalClass(nn.Module):
    def __init__(self,
                 bert_model: nn.Module,
                 classifier: nn.Module,
                 selection_encoder: nn.Module,
                 num_goals=6,
                 goal_tokens=False):
        super().__init__()

        self.bert_model = bert_model
        self.classifier = classifier
        self.goal_tokens = goal_tokens
        self.num_goals = num_goals
        self.selection_encoder = selection_encoder

    def forward(self,
                text, selections) -> Tensor: #batch x encoder hidden size #assumes that src is a batch in the style of a transformers.tokenization_utils_base.BatchEncoding
        """

        :param text: unstructured natural language strategy
        :param selections: selections made for the strategy
        :return: Final goal probabilities

        This function predicts the probability for the goal value existing in a bucket range for each goal
        """
        #calculate text embedding
        if self.bert_model != None:
            bert_output = self.bert_model(**text)
            if self.goal_tokens:
                bert_cls_embedding = bert_output[0][:, 1:self.num_goals+1,:]  # assumes that input is tokenized with <s> G1 G2 .... at the front, batch x num_goals x hidden size
            else:
                bert_cls_embedding = bert_output[0][:, 0, :] # batch x bert hidden size
        else:
            bert_cls_embedding = None

        #calculate selection embedding
        if self.selection_encoder != None:
            if len(selections.shape)==4:
                selections = selections.view(selections.shape[0],
                                         selections.shape[1] * selections.shape[2] * selections.shape[3])
            else:
                selections = selections.view(selections.shape[0],
                                         selections.shape[1] * selections.shape[2])
                
            selection_embedding = self.selection_encoder(selections.float()) #batch x selection hidden size
        else:
            selection_embedding = None

        if bert_cls_embedding != None and selection_embedding != None:
            if self.goal_tokens: #need to concatenate selections to each individual goal representation
                batch_size, num_goals, _ = bert_cls_embedding.size()
                selection_embedding = selection_embedding[:,None,:].expand(-1, self.num_goals, -1) #expand selection embedding to match dimensions of text embedding
                combined_embedding = torch.cat((bert_cls_embedding, selection_embedding), dim=2)
            else:
                combined_embedding = torch.cat((bert_cls_embedding, selection_embedding), dim=1)

            goal_class_probs = self.classifier(combined_embedding, self.goal_tokens)
        elif bert_cls_embedding != None and selection_embedding == None:
            goal_class_probs = self.classifier(bert_cls_embedding, self.goal_tokens)
        elif bert_cls_embedding == None and selection_embedding != None:
            if self.goal_tokens:
                selection_embedding = selection_embedding[:, None, :].expand(-1, self.num_goals, -1)  # expand selection embedding to be expected size
            goal_class_probs = self.classifier(selection_embedding, self.goal_tokens)
        else:
            assert False, "Neither text or selection encoders were active"

        return goal_class_probs

class BertToGoalValue(nn.Module):
    def __init__(self,
                 bert_model: nn.Module,
                 regressor: nn.Module,
                 num_goals=6,
                 per_goal_attention=False):
        super().__init__()

        self.bert_model = bert_model
        self.regressor = regressor
        self.per_goal_attention = per_goal_attention
        self.num_goals = num_goals

    def forward(self,
                src, selections) -> Tensor: #batch x encoder hidden size #assumes that src is a batch in the style of a transformers.tokenization_utils_base.BatchEncoding
        """

        :param src: unstructured natural language strategy
        :param selections: selections made for the strategy
        :return: predicts a single value for each goal

        This function is used to predict a single value for each goal. We generally use MSE loss for this

        """
        bert_output = self.bert_model(**src)
        # cls_hidden_state = bert_output[0][:,1,:] #batch size by sequence length by hidden size. I am uncertain why I was using the second hidden state and not the first
        # cls_hidden_state = bert_output[0][:,0,:]
        if self.per_goal_attention:
          cls_hidden_state = bert_output[0][:,0:self.num_goals,:] #assumes that input is tokenized with Goal1Tok Goal2Tok .... at the front
        else:
          cls_hidden_state = bert_output[0][:,0,:]
        goal_class_probs = self.regressor(cls_hidden_state, self.per_goal_attention)

        return goal_class_probs

#originally copied from the OpenNMT github repo at https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
#changed have been made to make their code applicable to our problem
class GoalLabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, num_goals, num_goal_buckets):
        super(GoalLabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing/num_goal_buckets
        one_hot = torch.full((num_goal_buckets, num_goals), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x num_goal_buckets x num_goals
        target (LongTensor): batch_size x num_goals
        """
        model_prob = self.one_hot.repeat(target.size(0), 1, 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction='sum')


