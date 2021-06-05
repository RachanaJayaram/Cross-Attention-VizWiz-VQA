"""Training module."""

import logging
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.loss_utils as loss_utils
import utils.train_utils as train_utils


class TrainClass:
    def __init__(
        self,
        model,
        train_configs_inp,
        save_folder,
        final_save_name,
        snapshot_path,
        logger,
    ):
        self.model = model
        self.train_configs = train_configs_inp
        self.save_folder = save_folder
        self.final_save_name = final_save_name
        self.logger = logger
        self.device = torch.device("cuda")

        train_utils.print_model(self.model, self.logger)
        logger.info("Train params:\t%s\n", self.train_configs)
        self.logger.info(
            "TRAINING PARAMETERS:\t"
            "optimizer: adamax\t"
            "base_learning_rate = %.8f,\t"
            "grad_clip=%.2f\n",
            self.train_configs.base_learning_rate,
            self.train_configs.grad_clip,
        )

        self.optimizer = torch.optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.train_configs.base_learning_rate,
        )
        if snapshot_path:
            self._load_model(snapshot_path)

        self.rat_for_epochs = train_utils.get_reattention_tradeoff_for_epochs(
            self.train_configs
        )
        self.logger.info(
            "Reattention tradeoffs for epochs : %s", self.rat_for_epochs
        )

        lr_for_epochs = train_utils.get_lr_for_epochs(self.train_configs)[
            self.train_configs.start_epoch :
        ]
        self.logger.info("LR for epochs : %s", lr_for_epochs)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (
                lr_for_epochs[epoch] / self.train_configs.base_learning_rate
            ),
        )

    def train(self, train_loader, eval_loader):
        for epoch in range(
            self.train_configs.start_epoch, self.train_configs.number_of_epochs
        ):
            self.logger.info(
                "Training For Epoch: %d\tLearning rate = %.4f\tReattention tradeoff = %.4f",
                epoch,
                self.scheduler.get_last_lr()[0],
                self.rat_for_epochs[epoch],
            )
            epoch_start_time = time.time()
            train_size = len(train_loader.dataset)

            total_loss, total_attention_loss, total_score = self._train_epoch(
                train_loader, self.rat_for_epochs[epoch]
            )

            # Update learning rate. Skip updating in the last iteration.
            if epoch != self.train_configs.number_of_epochs - 1:
                self.scheduler.step()

            total_loss /= train_size
            total_attention_loss /= train_size
            total_score = 100 * total_score / train_size
            eval_score = 0

            self.logger.info(
                "epoch %d,\t"
                "train_size: %d,\t"
                "time: %.2f,\t"
                "train_loss: %.2f\t"
                "attention_loss: %.4f\n"
                "SCORE: %.4f\n\n",
                epoch,
                train_size,
                time.time() - epoch_start_time,
                total_loss,
                total_attention_loss,
                total_score,
            )

            if epoch == self.train_configs.number_of_epochs - 1:
                self.logger.info("Saving model as %s", "final.pth")
                model_path = os.path.join(self.save_folder, "final")
                train_utils.save_model(
                    model_path, self.model, self.optimizer, epoch, total_score
                )
            self._save_model_if_eligible(epoch, total_score)
            if (
                eval_loader
                and total_score > self.train_configs.save_score_threshold
            ):
                self.model.train(False)
                self.logger.info("Threshold reached. Evaluating..")
                eval_score, _ = evaluate(self.model, eval_loader)
                self.model.train(True)
                self.logger.info("EVAL SCORE : %.4f\n\n", eval_score * 100)

    def _train_epoch(self, train_loader, reattention_tradeoff):
        total_loss = 0
        total_score = 0
        total_attention_loss = 0

        for _, (image_features, _, question, labels) in enumerate(
            tqdm(
                train_loader,
                total=len(train_loader),
                position=0,
                leave=True,
                colour="blue",
            )
        ):
            image_features = Variable(image_features).to(self.device)
            question = Variable(question).to(self.device)
            labels = Variable(labels).to(self.device)
            pred, v_att, v_re_att, _ = self.model(image_features, question)

            loss, att_loss = loss_utils.calculate_loss(
                pred,
                labels,
                self.model.module.reattention_added(),
                v_att,
                v_re_att,
                reattention_tradeoff,
            )

            # Clearing old gradients.
            self.optimizer.zero_grad()

            # Computes the gradient for the parameters.
            loss.backward()

            # Clips the norm of the overall gradient. Prevents exploding gradients.
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_configs.grad_clip
            )

            # Updates all the parameters based on the gradients.
            self.optimizer.step()

            total_loss += loss.data.item() * image_features.size(0)
            total_score += loss_utils.compute_score(pred, labels.data).sum()
            total_attention_loss += att_loss * image_features.size(0)

        return total_loss, total_attention_loss, total_score

    def _load_model(self, snapshot_path):
        model_data = torch.load(snapshot_path)
        self.model.load_state_dict(model_data.get("model_state", model_data))
        self.optimizer.load_state_dict(
            model_data.get("optimizer_state", model_data)
        )

        self.train_configs.start_epoch = model_data["epoch"] + 1

    def _save_model_if_eligible(self, epoch, total_score):
        if total_score >= 75 and epoch % self.train_configs.save_step == 0:
            save_name = "model_epoch{0}_score_{1}.pth".format(
                epoch, int(total_score)
            )
            self.logger.info("Saving model as %s", save_name)
            model_path = os.path.join(self.save_folder, save_name)
            train_utils.save_model(
                model_path, self.model, self.optimizer, epoch, total_score
            )


def train(
    model: nn.Module,
    train_configs: train_utils.TrainingConfigs,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    save_folder: str,
    final_save_name: str,
    snapshot_path: Optional[str],
    logger: logging.Logger,
):
    train_obj = TrainClass(
        model,
        train_configs,
        save_folder,
        final_save_name,
        snapshot_path,
        logger,
    )
    train_obj.train(train_loader, eval_loader)


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    device = torch.device("cuda")
    for _, (image_features, _, question, labels) in enumerate(
        tqdm(
            dataloader,
            total=len(dataloader),
            position=0,
            leave=True,
            colour="blue",
        )
    ):
        image_features = image_features.cuda()
        question = question.cuda()
        labels = labels.cuda()
        pred, _, _, _ = model(image_features, question)
        batch_score = loss_utils.compute_score(pred, labels).sum()
        score += batch_score
        upper_bound += (labels.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
