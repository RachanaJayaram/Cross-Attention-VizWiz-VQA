"""Module for extracting a feature vector from the input question."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.multi_layer_net import MultiLayerNet


class WordEmbedding(nn.Module):
    """A class for extracting feature vectors from tokens in the question.

    This class essentially behaves as a look-up table storing fixed-length
    300-dimensional word embeddings are extracted. Embeddings are first
    initialized using GloVe word embeddings,
    (https://nlp.stanford.edu/pubs/glove.pdf).
    The embeddings will be finetuned during training to so that they are more
    specific of the task at hand (https://arxiv.org/pdf/1505.07931.pdf)

    Attributes:
        embedding_lookup: Embedding table that returns vector embedding given
            an index. The table also doubles as a trainable layer.
    """

    def __init__(
        self,
        vocabulary_size: int,
        pretrained_vectors_file: str,
        embedding_dimension: int = 300,
        dropout: float = 0.0,
    ):
        """Initializes WordEmbedding.

        Args:
            vocabulary_size: Size of the lookup table.
            pretrained_vectors_file: Path to numpy file containing pretrained
                vector embeddings of all the words in the model's vocabulary.
                See tools/create_embedding.py to generate this file.
            embedding_dimension: Dimension of the extracted vector.
        """
        super().__init__()

        # padding_idx: dataset.py pads the list of token for shorter questions
        # with a value equal to the size of the vocabulary. Embeddings at an
        # index equal to this value (=vocabulary_size) are not updated during
        # training.
        self.embedding_lookup = nn.Embedding(
            num_embeddings=vocabulary_size + 1,
            embedding_dim=embedding_dimension,
            padding_idx=vocabulary_size,
        )

        self.dropout = nn.Dropout(dropout)

        # Ensures that the word vectors are fine tuned to the VQA task during
        # training.
        self.embedding_lookup.weight.requires_grad = True

        pretrained_weights = torch.from_numpy(np.load(pretrained_vectors_file))
        self.embedding_lookup.weight.data[:vocabulary_size] = pretrained_weights

    def forward(self, inp):
        """Defines the computation performed at every call."""
        return self.dropout(self.embedding_lookup(inp))


class QuestionEmbedding(nn.Module):
    """For extracting features from word indices of a tokenized question.

    The input list of word indices are w.r.t to the WordEmbedding lookup table.
    """

    def __init__(
        self,
        input_dimension: int,
        number_hidden_units: int,
        number_of_layers: int,
    ):
        """Initializes QuestionEmbedding.

        Args:
            input_dimension: The number of expected features in the input inp.
            number_hidden_units: The number of features in the hidden state h.
            number_of_layers: Number of recurrent layers.
        """
        super().__init__()

        self.number_hidden_units = number_hidden_units
        self.number_of_layers = number_of_layers
        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=number_hidden_units,
            num_layers=number_of_layers,
            bidirectional=False,
            batch_first=True,
        )

    def init_hidden(self, batch_size):
        """Grabs parameters of the model to instantiate a tensor on same device.

        Based on
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L56

        Args:
            bcz: batch size.
        """
        weight = next(self.parameters())
        return (
            weight.new_zeros(
                self.number_of_layers, batch_size, self.number_hidden_units
            ),
            weight.new_zeros(
                self.number_of_layers, batch_size, self.number_hidden_units
            ),
        )

    def forward(self, inp):
        """Defines the computation performed at every call.

        Args:
            inp:
                tensor containing the features of the input sequence:
                (batch, sequence, input_dimension)
                Tensor has shape:
                (batch_size, question_sequence_length, input_size).
        """
        batch_size = inp.size(0)
        hidden = self.init_hidden(batch_size)

        # Compact weights into single contiguous chunk of memory.
        self.lstm.flatten_parameters()

        output, hidden = self.lstm(inp, hidden)

        return output


class QuestionSelfAttention(nn.Module):
    def __init__(self, num_hid, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        self.W1_self_att_q = MultiLayerNet(
            [num_hid, num_hid], dropout=dropout, activation_fn_name=None
        )
        self.W2_self_att_q = MultiLayerNet(
            [num_hid, 1], activation_fn_name=None
        )

    def forward(self, ques_feat):
        """
        ques_feat: [batch, 14, num_hid]
        """
        batch_size = ques_feat.shape[0]
        q_len = ques_feat.shape[1]

        # (batch*14,num_hid)
        ques_feat_reshape = ques_feat.contiguous().view(-1, self.num_hid)
        # (batch, 14)
        atten_1 = self.W1_self_att_q(ques_feat_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, 1, 14)
        weight = F.softmax(atten.t(), dim=1).view(-1, 1, q_len)
        ques_feat_self_att = torch.bmm(weight, ques_feat)
        ques_feat_self_att = ques_feat_self_att.view(-1, self.num_hid)
        # (batch, num_hid)
        ques_feat_self_att = self.drop(ques_feat_self_att)
        return ques_feat_self_att
