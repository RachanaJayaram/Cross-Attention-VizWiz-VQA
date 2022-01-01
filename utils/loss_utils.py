"""Module for utilities that aide loss and score calculations."""

import torch
import torch.nn as nn


def classification_loss(logits, labels):
    """Binary cross entropy is used to calculate the classification loss.

    binary_cross_entropy_with_logits combines a Sigmoid layer and the BCELoss
    in one single class as it is more stable, which is why the sigmoid
    activation was ommitted in the classification layer.
    Ground truth soft scores for labels is calculated as per:
    https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/compute_softscore.py#L80
    and http://visualqa.org/evaluation.html
    Args:
        logits:
            Tensor containing predicted score for each answer candidiate
            predicted. Scores are pre-sigmoid and not between 0 and 1.
            Tensor is of dimension (batch size, answer vocabulary size).
        labels:
            Ground truth soft scores for each answer in the set of candidiate
            answers.
            Soft scores/ accuracies are between [0, 1].
            Tensor is of dimension (batch size, answer vocabulary size).

    Returns:
        Binary cross entropy loss between logits and labels after sigmoid
        activation.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def attention_consistency_loss(attention_weights, reattention_weights):
    """Returns Attention consistency loss.

    Attention consistency loss is defined as the distance between the visual
    attention maps learned in the attention layer (learned from the fusion of
    the question and image features) and the attention map learned in the
    re-attention layer(learned from the fusion of the gleaned answer
    representation and image features). It is implemented here as defined in
    'Re-attention for Visual Question Answering' by Guo et al
    (https://ojs.aaai.org//index.php/AAAI/article/view/5338). Attention
    consistency loss is used to re-learn the importance of the visual features
    guided by which objects correspond to the learned answer,

    Args:
        attention_weights: visual attention weights learned in the attention
            layer, prior to obtaining the answer representation.
        reattention_weights: visual attention weights learned in the
            re-attention layer.

    Returns:
        attention consistency loss.
    """
    return torch.sum(
        torch.square(torch.subtract(attention_weights, reattention_weights))
    )


def compute_score(logits, labels):
    """Computes score of the prediction.

    Ground truth soft scores for labels is calculated as per:
    https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/compute_softscore.py#L80
    and http://visualqa.org/evaluation.html
    Args:
        logits:
            Tensor containing predicted score for each answer candidiate
            predicted. Scores are pre-sigmoid and not between 0 and 1.
            Tensor is of dimension (batch size, answer vocabulary size).
        labels:
            Ground truth soft scores for each answer in the set of candidiate
            answers.
            Soft scores/ accuracies are between [0, 1].
            Tensor is of dimension (batch size, answer vocabulary size).
    """
    # logits has dimension [batch_size, num_ans_candidiates]
    # Each sample in the batch is of dimension num_ans_candidiates and each
    # element is the predicted score for the corresponding answer.

    # Finding the element index (offset) with max score for each sample in the
    # batch (i.e. argmax).
    # logits will now have dimension [batch_size] i.e. batch_size * 1.
    logits = torch.max(logits, 1)[1].data

    # Making zero vector of dimension [batch_size, num_ans_candidiates].
    one_hots = torch.zeros(*labels.size()).cuda()

    # Making a one-hot vector of dimension [batch_size, num_ans_candidiates].
    # For each sample the value at an index = index with max score, will be
    # set to 1.
    # This one hot vector is essentially encodes the most probable answer in
    # the vocabulary as predicted by the model.
    one_hots.scatter_(1, logits.view(-1, 1), 1)

    # Score will be a vector of dimension [batch_size, num_ans_candidiates].
    # Each sample will be a vector containing all zeroes, except at the index
    # of the predicted answer which contains the soft score/accuracy of that
    # answer calculated as per http://visualqa.org/evaluation.html.
    scores = one_hots * labels
    return scores
