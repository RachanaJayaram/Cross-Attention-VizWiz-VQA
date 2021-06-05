"Main module to be run for training the model."

import json

import torch
import torch.nn as nn
from absl import app
from torch.utils.data import DataLoader, ConcatDataset

import utils.dataset as dataset
import utils.train_utils as train_utils
from model.vqa_model import VQAModel, ModelParams
from train import train
from utils.flags import FLAGS


def main(_):
    """Main function"""
    train_utils.create_dir(FLAGS.save_folder)
    logger = train_utils.get_logger("VQA", FLAGS.save_folder)

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True

    data_params = json.load(open(FLAGS.data_params_path))
    dictionary = dataset.Dictionary.load_from_file(FLAGS.dictionary_path)
    train_configs = train_utils.TrainingConfigs(
        start_epoch=FLAGS.start_epoch,
        number_of_epochs=FLAGS.number_of_epochs,
        batch_size=FLAGS.batch_size,
        base_learning_rate=FLAGS.base_learning_rate,
        warmup_length=FLAGS.warmup_length,
        warmup_factor=FLAGS.warmup_factor,
        lr_decay_factor=FLAGS.lr_decay_factor,
        lr_decay_start=FLAGS.lr_decay_start,
        ra_decay_factor=FLAGS.ra_decay_factor,
        ra_decay_start=FLAGS.ra_decay_start,
        decay_step=FLAGS.decay_step,
        save_score_threshold=FLAGS.save_score_threshold,
        save_step=FLAGS.save_step,
        grad_clip=FLAGS.grad_clip,
        reattention_tradeoff=FLAGS.reattention_tradeoff,
    )

    model_params = ModelParams(
        add_self_attention=FLAGS.add_self_attention,
        add_reattention=FLAGS.add_reattention,
        fusion_method=FLAGS.fusion_method,
        question_sequence_length=dataset.MAX_QUES_SEQ_LEN,
        number_of_objects=dataset.NO_OBJECTS,
        word_embedding_dimension=data_params["word_feat_dimension"],
        object_embedding_dimension=data_params["image_feat_dimension"],
        vocabulary_size=data_params["vocabulary_size"],
        num_ans_candidates=data_params["number_of_answer_candidiates"],
    )
    logger.info("Model params:\t%s\n", model_params)

    model = VQAModel(
        glove_path=FLAGS.glove_path,
        model_params=model_params,
        hidden_dimension=FLAGS.hidden_dimension,
    ).cuda()
    model = nn.DataParallel(model).cuda()

    train_dset = dataset.VQAFeatureDataset(
        name="train",
        dictionary=dictionary,
    )

    eval_dset = dataset.VQAFeatureDataset("val", dictionary)

    if FLAGS.use_train_and_val:
        train_dset = ConcatDataset([train_dset, eval_dset])
        eval_loader = None

    else:
        eval_loader = DataLoader(
            eval_dset, FLAGS.batch_size, shuffle=True, num_workers=1
        )

    train_loader = DataLoader(
        train_dset,
        train_configs.batch_size,
        shuffle=True,
        num_workers=1,
    )

    train(
        model,
        train_configs,
        train_loader,
        eval_loader,
        FLAGS.save_folder,
        FLAGS.final_save_name,
        FLAGS.snapshot_path,
        logger,
    )


if __name__ == "__main__":
    app.run(main)
