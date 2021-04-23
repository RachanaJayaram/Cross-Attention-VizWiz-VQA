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

    def forward(self, x):
        logits = self.logits(x)
        w = nn.functional.softmax(logits, dim=1)
        return w

    def logits(self, x):
        similarity_matrix = self.dropout(x)
        logits = self.linear(similarity_matrix)
        return logits


class ReAttention(nn.Module):
    def __init__(self, hidden_dimension, number_of_objects, dropout=0.2):
        super().__init__()
        self.number_of_objects = number_of_objects

        self.projection_net = MultiLayerNet(
            [hidden_dimension * 3, hidden_dimension]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hidden_dimension, 1), dim=None)

    def forward(self, r, q_proj, v_proj):
        avg_q = torch.mean(q_proj, 1)

        question_embedded_answer = self.projection_net(
            torch.cat((r, avg_q), dim=1)
        )
        question_embedded_answer = question_embedded_answer.unsqueeze(1).repeat(
            1, self.number_of_objects, 1
        )

        joint_repr = question_embedded_answer * v_proj
        joint_repr = self.dropout(joint_repr)

        return self.linear(joint_repr)
