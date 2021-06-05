"""Module for the attention layer."""

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from utils.flags import FusionMethod
from model.multi_layer_net import MultiLayerNet


class Attention(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # self.att_w_1 = MultiLayerNet(
        #     [input_dimension, input_dimension],
        #     dropout=dropout,
        #     activation_fn_name=None,
        # )
        # self.att_w_2 = MultiLayerNet(
        #     [input_dimension, 1], activation_fn_name=None
        # )
        self.linear = weight_norm(nn.Linear(input_dimension, 1), dim=None)

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        similarity_matrix = self.dropout(inp)
        # w_1 = torch.tanh(self.att_w_1(similarity_matrix))
        # return self.att_w_2(w_1)
        logits = self.linear(similarity_matrix)
        return logits


class NewAttention(nn.Module):
    def __init__(self, input_dimension, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(1, 1), dim=None)

    def forward(self, inp):
        logits = self.logits(inp)
        return nn.functional.softmax(logits, dim=1)

    def logits(self, inp):
        similarity_matrix = self.dropout(inp)
        logits = self.linear(
            torch.max(similarity_matrix, dim=2, keepdim=True)[0]
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


class ReAttention(nn.Module):
    def __init__(
        self, hidden_dimension, number_of_objects, fusion_method, dropout=0.2
    ):
        super().__init__()
        self.number_of_objects = number_of_objects
        self.fusion_method = fusion_method

        self.non_linear_layer = None
        if self.fusion_method == FusionMethod.CONCAT:
            self.non_linear_layer = MultiLayerNet(
                [hidden_dimension * 2, hidden_dimension], dropout=0.2
            )

        self.dropout = nn.Dropout(dropout)

        self.linear = weight_norm(nn.Linear(hidden_dimension, 1), dim=None)

    def forward(self, r, q_proj, v_proj):
        answer_representation = r.unsqueeze(1).repeat(
            1, self.number_of_objects, 1
        )

        if self.fusion_method == FusionMethod.CONCAT:
            avg_q = torch.mean(q_proj, 1)
            avg_q = avg_q.unsqueeze(1).repeat(1, self.number_of_objects, 1)
            joint_input_features = torch.cat((v_proj, avg_q), dim=2)

        elif self.fusion_method == FusionMethod.HADAMARD:
            joint_input_features = v_proj

        joint_repr = answer_representation * joint_input_features

        if self.non_linear_layer:
            joint_repr = self.non_linear_layer(joint_repr)

        joint_repr = self.dropout(joint_repr)

        return nn.functional.softmax(self.linear(joint_repr), dim=1)
