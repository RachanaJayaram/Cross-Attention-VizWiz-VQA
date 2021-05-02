"""Module for a fully connected network used for non linear transformations."""

from typing import List, Optional

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class MultiLayerNet(nn.Module):
    """Simple class for a non-linear fully connect network."""

    def __init__(
        self,
        dimensions: List[int],
        activation_fn_name: Optional[str] = "ReLU",
        dropout: float = 0,
        add_last_activation: bool = True,
        bias: bool = True,
    ):
        """Initializes MultiLayerNet.

        Args:
            dimensions: List of neural net layer dimensions.
            activation_fn_name: Name of nn module for the activation function.
            dropout: Probablitity of dropout for regularization.
            add_last_activation: Determines whether or not the given activation
                function is applied after the last linear layer. Used in case
                a different activation function is required for the last layer.
            bias: Determines whether or not bias is added.
        """
        super().__init__()
        if activation_fn_name:
            activation_layer = getattr(nn, activation_fn_name)
        layers = list()
        for i in range(len(dimensions) - 1):
            if dropout:
                layers.append(nn.Dropout(dropout))

            # Reparameterization of the weight vectors for optimization.
            layers.append(
                weight_norm(
                    nn.Linear(dimensions[i], dimensions[i + 1], bias=bias),
                    dim=None,
                )
            )
            if i != (len(dimensions) - 2) and activation_fn_name:
                layers.append(activation_layer())

        if activation_fn_name and add_last_activation:
            layers.append(activation_layer())

        self.multi_layer_net = nn.Sequential(*layers)

    def forward(self, inp):
        """Defines the computation performed at every call."""
        return self.multi_layer_net(inp)
