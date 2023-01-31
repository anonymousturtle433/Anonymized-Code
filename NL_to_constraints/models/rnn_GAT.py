import torch
import torch.nn as nn
from NL_to_constraints.utils.constants import *
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, emb_weight_matrix=None,
                 bidirectional=False):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        if emb_weight_matrix is not None:
            self.embedding.load_state_dict({'weight': torch.tensor(emb_weight_matrix)})
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional)

        self.fc = nn.Linear(enc_hid_dim * 2, 2 * dec_hid_dim)

        self.dropout = nn.Dropout(dropout)
        assert enc_hid_dim == dec_hid_dim, "Both encoder hidden dimension and decoder hidden dimension should be equal"

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        # embedded = self.embedding(src)

        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class simpleDecoder(nn.Module):
    def __init__(self, hidden_dim, device, loss='CE'):
        super().__init__()

        self.hidden_size = hidden_dim
        self.device = device
        if loss == 'AXE':
            self.linear_layers = nn.ModuleList(
                [nn.Linear(self.encoder.enc_hid_dim * 2, self.decoder_gat.output_dim * self.decoder_gat.value_dim + 1)
                 for i in range(8)])
        else:
            self.constraint_heads = nn.ModuleList(
                [nn.Linear(self.hidden_size, len(CONSTRAINT_TYPES))
                 for i in range(8)])
            self.value_heads = nn.ModuleList(
                [nn.Linear(self.hidden_size, len(VALUE_TYPES))
                 for i in range(8)])
        # self.norm1 = nn.LayerNorm(self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, trg, hidden, encoder_outputs=None):
        _, output_logits, outputs_c, outputs_v = self.constraint_forward(trg, hidden, encoder_outputs)

        return None, output_logits, outputs_c, outputs_v

    def constraint_forward(self, trg, hidden, encoder_outputs=None):
        # hidden_outputs: [batch_size x self.hidden_dim] or list 8 * [batch_size x self.hidden_dim] if con-tokens is true
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        if not isinstance(hidden, list):
            hidden = hidden.unsqueeze(1)
        output_vectors = []
        # output_vectors: list of 8 [batch_size x 1 x (constraint_dim * value_dim)] tensors
        for i in range(max_len):
            if isinstance(hidden, list):
                input = hidden[i].unsqueeze(1)
            else:
                input = hidden
            constraint_logits = self.constraint_heads[i](input)
            value_logits = self.value_heads[i](input)
            constraint_logits = constraint_logits.permute(0, 2, 1)
            outputs = torch.matmul(constraint_logits, value_logits)
            outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2])

            outputs = outputs.unsqueeze(1)
            output_vectors.append(outputs)

        output_logits = torch.cat(output_vectors, dim=1)
        return None, output_logits, None, None


class Decoder(nn.Module):
    def __init__(self, output_dim, value_dim, dec_hid_dim, feedforward_dim, num_heads, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.value_dim = value_dim
        self.num_constraints = 8

        self.dec_hid_dim = dec_hid_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim

        self.gat = nn.MultiheadAttention(self.dec_hid_dim, self.num_heads)

        # self.linear1 = nn.Linear(self.dec_hid_dim, self.feedforward_dim)
        # self.linear2 = nn.Linear(self.dec_hid_dim + self.feedforward_dim, self.feedforward_dim)
        self.linear1_embed = nn.ModuleList(
            [nn.Linear(self.dec_hid_dim, self.feedforward_dim) for i in range(8)])

        self.linear2_embed = nn.ModuleList(
            [nn.Linear(self.dec_hid_dim, self.feedforward_dim) for i in range(8)])

        self.linear_con = nn.ModuleList(
            [nn.Linear(self.feedforward_dim, self.output_dim) for i in range(8)]
        )

        self.linear_val = nn.ModuleList(
            [nn.Linear(self.feedforward_dim, self.value_dim) for i in range(8)]
        )

        # self.linear_comb = nn.Linear(self.feedforward_dim, self.output_dim * self.value_dim + 1)
        self.norm1 = nn.LayerNorm(self.dec_hid_dim)
        self.norm2 = nn.LayerNorm(self.dec_hid_dim + self.feedforward_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

    def forward(self, trg, multihead_outputs):
        # multihead_outputs: [max_num_constraints (8) x batch_size x decoder_hid_dim]
        gat_outputs = self.gat(multihead_outputs, multihead_outputs, multihead_outputs)[0]
        # gat_outputs: [max_num_constraints (8) x batch_size x decoder_hid_dim]
        src = multihead_outputs + self.dropout(gat_outputs)
        src = self.norm1(src)

        total = []
        for c in range(self.num_constraints):
            out = self.relu(self.linear1_embed[c](src[c,:,:]))
            total.append(out.unsqueeze(0))
        con_inter = torch.cat(total)
        # con_inter: [num_constraints (8) x batch_size x feedforward_dim]

        logits = []
        for c in range(self.num_constraints):
            out = self.linear_con[c](self.dropout(con_inter[c,:,:]))
            logits.append(out.unsqueeze(0))
        constraint_logits = torch.cat(logits)
        # constraint_logits: [num_constraints (8) x batch_size x output_dim (10)]

        # val_src = self.norm2(torch.cat((src, con_inter), dim=2))
        # val_src: [num_constraints(8) x batch_size x (feedforward_dim + hidden_dim)]

        logits = []
        for c in range(self.num_constraints):
            out = self.linear_val[c](self.dropout(self.relu(self.linear2_embed[c](src[c,:,:]))))
            logits.append(out.unsqueeze(0))
        value_logits = torch.cat(logits)
        # value_logits: [num_constraints (8) x batch_size x value_dim]

        constraint_logits = constraint_logits.unsqueeze(3)
        # NEXT TWO LINES REQUIRED FOR AXE LOSS FUNCTION
        # output_combined = self.linear_comb(self.dropout(con_inter))
        # output_combined = output_combined.permute(1,0,2)
        # output_combined: [batch_size x max_num_constraints(8) x (output_dim * value_dim)]

        value_logits = value_logits.unsqueeze(2)
        constraint_logits = constraint_logits.permute(1, 0, 2, 3)
        value_logits = value_logits.permute(1, 0, 2, 3)
        output_logits = torch.matmul(constraint_logits, value_logits)
        # output_logits: [max_num_constraints(8) x batch_size x output_dim x value_dim]
        output_logits = output_logits.view(output_logits.shape[0], output_logits.shape[1],
                                           output_logits.shape[2] * output_logits.shape[3])
        constraint_logits = constraint_logits.squeeze()
        value_logits = value_logits.squeeze()

        # con_inter = self.relu(self.linear1(src))
        # # con_inter: [max_num_constraints(8) x batch_size x feedforward_dim]
        # constraint_logits = self.linear_con(self.dropout(con_inter))
        # # constraint_logits: [max_num_constraints(8) x batch_size x output_dim]
        # # MAKE BATCH_SIZE FIRST DIMENSION
        # # SPLIT INTO 8 linear layers similar to regular classification heads
        # val_src = self.norm2(torch.cat((src, con_inter), dim=2))
        # # val_src: [max_num_constraints(8) x batch_size x (feedforward_dim + decoder_hid_dim)]
        #
        # value_logits = self.linear_val(self.dropout(self.relu(self.linear2(val_src))))
        # # value_logits: [max_num_constraints(8) x batch_size x value_dim]
        # constraint_logits = constraint_logits.unsqueeze(3)
        # # NEXT TWO LINES REQUIRED FOR AXE LOSS FUNCTION
        # # output_combined = self.linear_comb(self.dropout(con_inter))
        # # output_combined = output_combined.permute(1,0,2)
        # # output_combined: [batch_size x max_num_constraints(8) x (output_dim * value_dim)]
        #
        # value_logits = value_logits.unsqueeze(2)
        # constraint_logits = constraint_logits.permute(1, 0, 2, 3)
        # value_logits = value_logits.permute(1, 0, 2, 3)
        # output_logits = torch.matmul(constraint_logits, value_logits)
        # # output_logits: [max_num_constraints(8) x batch_size x output_dim x value_dim]
        # output_logits = output_logits.view(output_logits.shape[0], output_logits.shape[1],
        #                                    output_logits.shape[2] * output_logits.shape[3])
        # constraint_logits = constraint_logits.squeeze()
        # value_logits = value_logits.squeeze()

        return gat_outputs, output_logits, constraint_logits, value_logits




class simpleNetworkwAttn(nn.Module):
    def __init__(self, encoder, device, loss='CE', test=False, attention=True):
        super().__init__()

        self.encoder = encoder
        self.device = device
        self.test_flag = test
        self.penalty = torch.nn.Parameter(torch.Tensor([1]))
        self.attention = attention
        if loss == 'AXE':
            self.linear_layers = nn.ModuleList(
                [nn.Linear(self.encoder.enc_hid_dim * 2, len(CONSTRAINT_TYPES) * len(VALUE_TYPES) + 1) for i in
                 range(8)])
        else:
            self.embed_layers = nn.ModuleList(
                [nn.MultiheadAttention(self.encoder.enc_hid_dim * 2, 1) for i in range(8)])
            self.constraint_heads = nn.ModuleList(
                [nn.Linear(self.encoder.enc_hid_dim * 2, len(CONSTRAINT_TYPES))
                 for i in range(8)])
            self.value_heads = nn.ModuleList(
                [nn.Linear(self.encoder.enc_hid_dim * 2, len(VALUE_TYPES))
                 for i in range(8)])
        self.norm1 = nn.LayerNorm(self.encoder.enc_hid_dim * 2)
        self.relu = nn.ReLU()

    def forward(self, src, trg):
        # encoder_outputs: [seq_len x batch_size x 2*dec_hidden_dim]
        # hidden_outputs: [batch_size x 2*dec_hidden_dim]
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        _, output_logits, outputs_c, outputs_v = self.greedy_decode(trg, hidden, encoder_outputs)

        return output_logits, outputs_c, outputs_v

    def greedy_decode(self, trg, hidden, encoder_outputs):
        # encoder_outputs: [seq_len x batch_size x 2*dec_hidden_dim]
        # hidden_outputs: [batch_size x 2*dec_hidden_dim]
        batch_size = trg.shape[0]
        max_len = trg.shape[1]

        hidden = hidden.unsqueeze(1)
        output_vectors = []

        # output_vectors: list of 8 [batch_size x 1 x (constraint_dim * value_dim)] tensors
        for i in range(max_len):
            attn_outputs = self.embed_layers[i](hidden, encoder_outputs, encoder_outputs)[0]
            # constraint_logits = self.constraint_heads[i](self.norm1(attn_outputs))
            # value_logits = self.value_heads[i](self.norm1(attn_outputs))
            constraint_logits = self.constraint_heads[i](attn_outputs)
            value_logits = self.value_heads[i](attn_outputs)
            constraint_logits = constraint_logits.permute(0, 2, 1)
            outputs = torch.matmul(constraint_logits, value_logits)
            outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2])

            outputs = outputs.unsqueeze(1)
            output_vectors.append(outputs)

        output_logits = torch.cat(output_vectors, dim=1)
        return None, output_logits, None, None


class Seq2Con(nn.Module):
    def __init__(self, encoder, decoder, device, test=False, enc_dec_attn=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.test_flag = test
        self.norm1 = nn.LayerNorm(self.encoder.enc_hid_dim * 2)
        self.enc_dec_attn = enc_dec_attn
        if type(self.decoder).__name__ == 'Decoder':
            self.embed_layers = nn.ModuleList([nn.MultiheadAttention(self.encoder.enc_hid_dim * 2, 1) for i in range(8)])
            self.linear_embed = nn.ModuleList(
                [nn.Linear(self.encoder.enc_hid_dim * 2, self.decoder.dec_hid_dim) for i in range(8)])

    def forward(self, src, trg):
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        if type(self.decoder).__name__ == 'simpleDecoder':
            _, output_logits, outputs_c, outputs_v = self.forward_simple(trg, hidden, encoder_outputs)
        elif type(self.decoder).__name__ == 'Decoder':
            _, output_logits, outputs_c, outputs_v = self.forward_gat(trg, hidden, encoder_outputs)
        # _, output_logits, outputs_c, outputs_v = self.greedy_decode(trg, hidden, encoder_outputs)

        return output_logits, outputs_c, outputs_v

    def forward_gat(self, trg, hidden, encoder_outputs):
        # encoder_outputs: [seq_len x batch_size x 2*dec_hidden_dim]
        # hidden_outputs: [batch_size x 2*dec_hidden_dim]
        batch_size = trg.shape[0]
        max_len = trg.shape[1]

        hidden = hidden.unsqueeze(0)
        constraint_vectors = []
        # constraint_vectors: list of 8 [1 x batch_size x dec_hidden_dim] tensors
        for i in range(max_len):
            if self.enc_dec_attn:
                # constraint_vectors.append(self.linear_embed[i](self.norm1(self.embed_layers[i](hidden, encoder_outputs, encoder_outputs)[0])))
                constraint_vectors.append(
                    self.linear_embed[i](self.embed_layers[i](hidden, encoder_outputs, encoder_outputs)[0]))
            else:
                constraint_vectors.append(
                    self.linear_embed[i](hidden))

        # Concatenate constraint embeddings
        constraint_vectors = torch.cat(constraint_vectors)
        gat_outputs, output_logits, constraint_logits, value_logits = self.decoder(trg, constraint_vectors)

        return gat_outputs, output_logits, constraint_logits, value_logits

    def forward_simple(self, trg, hidden, encoder_outputs = None):
        _, output_logits, outputs_c, outputs_v = self.decoder(trg, hidden)

        return _, output_logits, outputs_c, outputs_v

class BertToConstraintClass(nn.Module):
    def __init__(self,
                 bert_model: nn.Module,
                 decoder: nn.Module,
                 hidden_size: int,
                 con_tokens: bool = False,
                 loss='CE'):
        super().__init__()

        self.bert_model = bert_model
        # REPLACE WITH BERT SIZE
        self.decoder = decoder
        self.con_tokens = con_tokens
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(self.hidden_size)
        if type(self.decoder).__name__ == 'Decoder':
            self.linear_embed = nn.ModuleList(
                [nn.Linear(self.hidden_size, self.decoder.dec_hid_dim) for i in range(8)])
        self.relu = nn.ReLU()

    def forward(self,
                src, trg) -> Tensor:
        bert_output = self.bert_model(**src)

        cls_hidden_state = bert_output[0][:, 0, :]
        if self.con_tokens:
            constraint_hidden_states = []
            for i in range(8):
                constraint_hidden_states.append(bert_output[0][:, i + 1, :])
            cls_hidden_state = constraint_hidden_states
        if type(self.decoder).__name__ == 'simpleDecoder':
            _, output_logits, outputs_c, outputs_v = self.forward_simple(trg, cls_hidden_state)
        elif type(self.decoder).__name__ == 'Decoder':
            _, output_logits, outputs_c, outputs_v = self.forward_gat(trg, cls_hidden_state)
        return output_logits, None, None

    def forward_gat(self, trg, hidden):
        # encoder_outputs: [seq_len x batch_size x 2*dec_hidden_dim]
        # hidden_outputs: [batch_size x 2*dec_hidden_dim]
        batch_size = trg.shape[0]
        max_len = trg.shape[1]

        hidden = hidden.unsqueeze(0)
        constraint_vectors = []
        # constraint_vectors: list of 8 [1 x batch_size x dec_hidden_dim] tensors
        for i in range(max_len):
            constraint_vectors.append(
                self.linear_embed[i](hidden))

        # Concatenate constraint embeddings
        constraint_vectors = torch.cat(constraint_vectors)
        gat_outputs, output_logits, constraint_logits, value_logits = self.decoder(trg, constraint_vectors)

        return gat_outputs, output_logits, constraint_logits, value_logits

    def forward_simple(self, trg, cls_hidden_state):
        _, output_logits, outputs_c, outputs_v = self.decoder(trg, cls_hidden_state)

        return _, output_logits, outputs_c, outputs_v



