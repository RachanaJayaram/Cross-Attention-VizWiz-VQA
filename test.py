"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from absl import app
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.dataset as dataset
import utils.loss_utils as loss_utils
import utils.train_utils as train_utils
from model.vqa_model import ModelParams, VQAModel
from utils.dataset import Dictionary, VQAFeatureDataset
from utils.flags import FLAGS

SPLIT = "test"


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(
            dictionary.idx2word[q[i]]
            if q[i] < len(dictionary.idx2word)
            else "_"
        )
    return " ".join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    im_ids = [""] * N
    device = torch.device("cuda")
    idx = 0
    for i, (image_features, _, question, image_ids) in enumerate(
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
        logits, _, _ = model(image_features, question)
        pred[idx : idx + FLAGS.batch_size, :].copy_(logits.data)
        im_ids[idx : idx + FLAGS.batch_size] = list(image_ids) + [""] * (
            FLAGS.batch_size - len(image_ids)
        )
        idx += FLAGS.batch_size

    return pred, im_ids


def make_json(logits, im_ids, dataloader):
    results = []
    for i in range(logits.size(0)):
        result = {}
        if len(im_ids[i]) == 0:
            continue
        result["image"] = im_ids[i] + ".jpg"
        result["answer"] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def main(_):
    train_utils.create_dir(FLAGS.save_folder)
    logger = train_utils.get_logger("VQA", FLAGS.save_folder)

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True

    data_params = json.load(open(FLAGS.data_params_path))
    dictionary = dataset.Dictionary.load_from_file(FLAGS.dictionary_path)
    eval_dset = VQAFeatureDataset(SPLIT, dictionary)

    model_params = ModelParams(
        add_self_attention=FLAGS.add_self_attention,
        fusion_method=FLAGS.fusion_method,
        question_sequence_length=dataset.MAX_QUES_SEQ_LEN,
        number_of_objects=dataset.NO_OBJECTS,
        word_embedding_dimension=data_params["word_feat_dimension"],
        object_embedding_dimension=data_params["image_feat_dimension"],
        vocabulary_size=data_params["vocabulary_size"],
        num_ans_candidates=data_params["number_of_answer_candidiates"],
    )

    model = VQAModel(
        glove_path=FLAGS.glove_path,
        model_params=model_params,
        hidden_dimension=FLAGS.hidden_dimension,
    ).cuda()
    model = nn.DataParallel(model).cuda()
    model.train(False)
    eval_loader = DataLoader(
        eval_dset,
        FLAGS.batch_size,
        shuffle=False,
        num_workers=1,
    )

    def process(model, eval_loader):
        model_path = FLAGS.snapshot_path
        print("loading %s" % model_path)

        model_data = torch.load(model_path)
        model.load_state_dict(model_data.get("model_state", model_data))
        model.train(False)

        logits, im_ids = get_logits(model, eval_loader)
        results = make_json(logits, im_ids, eval_loader)

        train_utils.create_dir(FLAGS.save_folder)
        with open(FLAGS.save_folder + "/%s.json" % (SPLIT), "w") as f:
            json.dump(results, f)

    process(model, eval_loader)


if __name__ == "__main__":
    app.run(main)
