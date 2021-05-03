import os
import sys
import h5py

import _pickle as cPickle
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import Dictionary


MAX_QUES_SEQ_LEN = 26
NO_OBJECTS = 36
URL_FEATURE_SERVER = "http://127.0.0.1:6000/GetFeature"


class VQAFeatureDataset(Dataset):
    def __init__(self, dataroot="data"):
        super(VQAFeatureDataset, self).__init__()
        self.dictionary = Dictionary.load_from_file(
            os.path.join(dataroot, "glove", "dictionary.pkl")
        )
        ans2label_path = os.path.join(
            dataroot,
            "cache",
            "trainval_ans2label.pkl",
        )
        label2ans_path = os.path.join(
            dataroot,
            "cache",
            "trainval_label2ans.pkl",
        )
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_ans_candidates = len(self.ans2label)

        name = "demo"
        self.img_id2idx = cPickle.load(
            open(
                os.path.join(
                    dataroot,
                    "imgids/%s%s_imgid2idx.pkl" % (name, 36),
                ),
                "rb",
            )
        )
        h5_dataroot = dataroot + "/Bottom-up-features-fixed"

        h5_path = os.path.join(h5_dataroot, "%s%s.hdf5" % (name, "36"))

        print("loading features from h5 file %s" % h5_path)
        hf_file = h5py.File(h5_path, "r")
        self.features = hf_file.get("image_features")
        self.bboxes = hf_file.get("image_bb")
        self.question = None
        self.image_id = None

    def set_input(self, image_id, question):
        tokens = self.dictionary.tokenize(question, False)
        tokens = tokens[:MAX_QUES_SEQ_LEN]
        if len(tokens) < MAX_QUES_SEQ_LEN:
            padding = [self.dictionary.padding_idx] * (
                MAX_QUES_SEQ_LEN - len(tokens)
            )
            tokens = tokens + padding
        self.question = torch.from_numpy(np.array(tokens))
        self.image_id = image_id

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[self.img_id2idx[self.image_id]]),
            torch.from_numpy(self.bboxes[self.img_id2idx[self.image_id]]),
            self.question,
        )

    def __len__(self):
        return 1
