"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import math

import numpy as np
import torch
from torch.autograd import Variable


def torch_broadcast_adj_matrix(
    adj_matrix, label_num=11, device=torch.device("cuda")
):
    """broudcast spatial relation graph

    Args:
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:
        result: [batch_size,num_boxes, num_boxes, label_num]
    """
    result = []
    for i in range(1, label_num + 1):
        index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze()
        curr_result = torch.zeros(
            adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2]
        )
        curr_result = curr_result.view(-1)
        curr_result[index] += 1
        result.append(
            curr_result.view(
                (
                    adj_matrix.shape[0],
                    adj_matrix.shape[1],
                    adj_matrix.shape[2],
                    1,
                )
            )
        )
    result = torch.cat(result, dim=3)
    return result


def prepare_spatial_matrix(spa_adj_matrix, num_objects, spa_label_num, device):

    spa_adj_matrix = spa_adj_matrix.to(device)
    spa_adj_matrix = spa_adj_matrix[:, :num_objects, :num_objects]
    spa_adj_matrix = torch_broadcast_adj_matrix(
        spa_adj_matrix, label_num=spa_label_num, device=device
    )
    spa_adj_matrix_var = Variable(spa_adj_matrix).to(device)
    return spa_adj_matrix_var
