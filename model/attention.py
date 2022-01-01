"""Module for the attention layer."""

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from model.multi_layer_net import MultiLayerNet


class Attention(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(input_dimension, 1), dim=None)

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        similarity_matrix = self.dropout(inp)
        logits = self.linear(similarity_matrix)
        return logits


class NewAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(1, 1), dim=None)

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        similarity_matrix = self.dropout(inp)
        logits = self.linear(
            torch.avg(similarity_matrix, dim=2, keepdim=True)[0]
        )
        return logits


class SelfAttention(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_att_w_1 = MultiLayerNet(
            [input_dimension, input_dimension],
            dropout=dropout,
            activation_fn_name=None,
        )
        self.self_att_w_2 = MultiLayerNet(
            [input_dimension, 1], activation_fn_name=None
        )

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        inp = self.dropout(inp)
        w_1 = torch.tanh(self.self_att_w_1(inp))
        return self.self_att_w_2(w_1)
