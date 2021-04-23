"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.graph.graph_att import GAttNet as GAT
from model.multi_layer_net import MultiLayerNet
from model.question_embedding import QuestionSelfAttention


def q_expand_v_cat(q, v, mask=True):
    q = q.view(q.size(0), 1, q.size(1))
    repeat_vals = (-1, v.shape[1], -1)
    q_expand = q.expand(*repeat_vals).clone()
    if mask:
        v_sum = v.sum(-1)
        mask_index = torch.nonzero(v_sum == 0)
        if mask_index.dim() > 1:
            q_expand[mask_index[:, 0], mask_index[:, 1]] = 0
    v_cat_q = torch.cat((v, q_expand), dim=-1)
    return v_cat_q


class ExplicitRelationEncoder(nn.Module):
    def __init__(
        self,
        v_dim,
        q_dim,
        out_dim,
        dir_num,
        label_num,
        nongt_dim=20,
        num_heads=16,
        num_steps=1,
        residual_connection=True,
        label_bias=True,
    ):
        super(ExplicitRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        self.num_steps = num_steps
        self.residual_connection = residual_connection
        print(
            "In ExplicitRelationEncoder, num of graph propogation steps:",
            "%d, residual_connection: %s"
            % (self.num_steps, self.residual_connection),
        )

        if self.v_dim != self.out_dim:
            self.v_transform = MultiLayerNet([v_dim, out_dim])
        else:
            self.v_transform = None
        in_dim = out_dim + q_dim
        self.explicit_relation = GAT(
            dir_num,
            label_num,
            in_dim,
            out_dim,
            nongt_dim=nongt_dim,
            num_heads=num_heads,
            label_bias=label_bias,
            pos_emb_dim=-1,
        )

    def forward(self, v, exp_adj_matrix, q):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q: [batch_size, q_dim]
            exp_adj_matrix: [batch_size, num_rois, num_rois, num_labels]

        Returns:
            output: [batch_size, num_rois, out_dim]
        """
        exp_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            v_cat_q = q_expand_v_cat(q, exp_v, mask=True)
            exp_v_rel = self.explicit_relation.forward(v_cat_q, exp_adj_matrix)
            if self.residual_connection:
                exp_v += exp_v_rel
            else:
                exp_v = exp_v_rel
        return exp_v


class RelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, hidden_dimension, attention_heads):
        super().__init__()
        self.question_self_attention_net = QuestionSelfAttention(
            hidden_dimension, 0.2
        )
        self.encoder_net = ExplicitRelationEncoder(
            v_dim=v_dim,
            q_dim=q_dim,
            out_dim=hidden_dimension,
            dir_num=2,
            label_num=11,
            nongt_dim=20,
            num_heads=attention_heads,
            num_steps=1,
            residual_connection=True,
            label_bias=True,
        )

    def forward(self, v, q, spa_adj_matrix):
        q_emb_self_att = self.question_self_attention_net(q)
        return self.encoder_net.forward(v, spa_adj_matrix, q_emb_self_att)
