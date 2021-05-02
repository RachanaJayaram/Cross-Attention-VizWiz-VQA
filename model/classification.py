"""Module for classification.

The loss layer consists of two branches. One of them is the classification
branch which is used to determine the accurate answer. The other branch is the
re-attention procedure, which utilizes answer representations to guide visual
importance learning -> (https://ojs.aaai.org//index.php/AAAI/article/view/5338)
"""

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Classifier(nn.Module):
    """Classifies the obtained joint embedding/answer representation.

    This multi label classifier projects the joint embedding to a dimension
    equal to the number of answer candidiates using a non-linear layer. The
    obtained tensor is then passed through a linear layer to obtain a weighted
    representation.

    The score for each label/answer is obtained by passing the above tensor
    through a sigmoid function (s = sigmoid(w_0 * f_0(h)). The sigmoid
    activation is applied in a 'classification_loss' method outside the class).
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float = 0,
    ):
        """Initializes Classifier.

        Args:
            input_dimension: The input dimension of the input.
            hidden_dimension: The dimension of the intermediate hidden layer.
            output_dimension: The required dimension for the output.
            dropout: Probability of dropout for regularization.
        """
        super().__init__()
        layers = [
            weight_norm(nn.Linear(input_dimension, hidden_dimension), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(
                nn.Linear(hidden_dimension, output_dimension), dim=None
            ),
        ]
        self.classifier_net = nn.Sequential(*layers)

    def forward(self, inp):
        """Defines the computation performed at every call."""
        return self.classifier_net(inp)
