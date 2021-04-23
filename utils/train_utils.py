"""Module for utilities that aide model training."""

import errno
import functools
import logging
import operator
import os
from typing import List

import attr
import torch


@attr.s
class TrainingConfigs:  # pylint: disable=too-few-public-methods
    """Stores configurations specific to training.

    Attributes:
        start_epoch(int): Epoch at which training should start/restart.
        number_of_epochs (int): Number of epochs for training.
        batch_size (int): The number of training examples in one forward/
            backward pass.
        base_learning_rate (float): The default rate at which weights are
            updated during training.
        warmup_length (int): Number of epochs for the warmup stage.
        warmup_factor (float): Factor by which lr is multiplied.
        decay_factor (float): Decay factor at which the lr is reduced.
        decay_start (int): Epoch at which learning rate decay should start.
        decay_step (int): Determines epochs for which decay is skipped.
        save_score_threshold (float): Threshold for model score after which
            the models are saved.
        save_step (int): Determines epochs for which the model is saved after
            the score threshold is reached.
        grad_clip (float): Max norm of the gradients for grad clipping.
    """

    start_epoch: int = attr.ib()
    number_of_epochs: int = attr.ib()
    batch_size: int = attr.ib()
    base_learning_rate: float = attr.ib()
    warmup_length: int = attr.ib()
    warmup_factor: float = attr.ib()
    decay_factor: float = attr.ib()
    decay_start: int = attr.ib()
    decay_step: int = attr.ib()
    save_score_threshold: float = attr.ib()
    save_step: int = attr.ib()
    grad_clip: float = attr.ib(default=0.25)


def get_lr_for_epochs(training_configs: TrainingConfigs) -> List[float]:
    """Get a list containing learning rate for each epoch.

    Args:
        training_configs: Populated instance of TrainingConfigs.

    Returns:
        list containing learning rate values corresponding to each epoch index.
    """
    lr_for_epochs = list()  # type: List[float]

    # Have sharp increase in learning rates for initial epochs.
    lr_for_epochs.extend(
        list(
            training_configs.warmup_factor
            * _epoch_i
            * training_configs.base_learning_rate
            for _epoch_i in range(1, 1 + training_configs.warmup_length)
        )
    )

    # Set default lr for non-warmup epochs.
    lr_for_epochs.extend(
        [training_configs.base_learning_rate]
        * (training_configs.decay_start - training_configs.warmup_length)
    )

    _decay_epochs = list(
        range(
            training_configs.decay_start,
            training_configs.number_of_epochs,
            training_configs.decay_step,
        )
    )

    current_lr = training_configs.base_learning_rate

    for _epoch_i in range(
        training_configs.decay_start, training_configs.number_of_epochs
    ):
        if _epoch_i in _decay_epochs:
            current_lr *= training_configs.decay_factor
        lr_for_epochs.append(current_lr)

    return lr_for_epochs


def create_dir(path: str) -> None:
    """Creates directory at given path if it doesnt exist.

    Taken from: https://github.com/jnhwkim/ban-vqa/blob/demo/utils.py

    Args:
        path: String path to directory.

    Returns: None.

    Raises:
        Error: Generic error.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def get_logger(logger_name="VQA", log_directory=None):
    """Create and configure a logger instance."""

    # create logger for prd_ci
    logger = logging.getLogger(logger_name)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%I:%M:%S %p",
    )

    # create file handler for logger.
    if log_directory:
        file_handler = logging.FileHandler(
            "{}/{}".format(log_directory, "info.log"), "a"
        )
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

    cl_handler = logging.StreamHandler()
    cl_handler.setLevel(level=logging.INFO)
    cl_handler.setFormatter(formatter)

    # add handlers to logger.
    logger.addHandler(cl_handler)
    if log_directory:
        logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def print_model(model, logger):
    """Displays model and number of parameters.

    Taken from: https://github.com/jnhwkim/ban-vqa/blob/demo/utils.py
    """
    print(model)
    number_of_parameters = 0
    for weights in model.parameters():
        number_of_parameters += functools.reduce(
            operator.mul, weights.size(), 1
        )
    logger.info("number_of_parameters=%d", number_of_parameters)


def save_model(path, model, optimizer, epoch, score):
    """Saves model at given path.

    Taken from: https://github.com/jnhwkim/ban-vqa/blob/demo/utils.py
    """
    model_dict = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "score": score,
    }
    model_dict["optimizer_state"] = optimizer.state_dict()

    torch.save(model_dict, path)
