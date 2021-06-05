import datetime
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.flags import FLAGS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vqa_model import ModelParams, VQAModel
import demo.demo_dataset as dataset
import demo.visualize as visualize


class Inference:
    def __init__(self):
        self.model = self._load_model()
        self.demo_data = dataset.VQAFeatureDataset()

    def _get_answer(self, p, dataloader):
        _m, idx = p.max(1)
        return dataloader.dataset.label2ans[idx.item()]

    def _load_model(self):
        data_params = json.load(open(FLAGS.data_params_path))
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
        model = VQAModel(
            glove_path=FLAGS.glove_path,
            model_params=model_params,
            hidden_dimension=FLAGS.hidden_dimension,
        ).cuda()
        FLAGS.snapshot_path = (
            "/home/rachana/Documents/vizwiz/save_folder/self_cross_3/50"
        )
        model_path = FLAGS.snapshot_path
        print("loading %s" % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get("model_state", model_data))
        model.train(False)
        return model

    def get_prediction(self, image_id, question, batch_size=1):
        self.demo_data.set_input(image_id, question)
        demo_data_loader = DataLoader(
            self.demo_data,
            batch_size,
            shuffle=False,
            num_workers=1,
        )
        visual_feature, bboxes, question = iter(demo_data_loader).next()
        visual_feature = Variable(visual_feature).cuda()
        bboxes = Variable(bboxes).cuda()
        question = Variable(question).cuda()
        pred, i_att, _, q_att = self.model(visual_feature, question)
        answer = self._get_answer(pred.data, demo_data_loader)

        return (
            answer,
            i_att,
            q_att,
            bboxes,
        )
