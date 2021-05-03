"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function

import json
import os

import _pickle as cPickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_QUES_SEQ_LEN = 26
NO_OBJECTS = 36


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers["labels"]:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = (
            sentence.replace(",", "").replace("?", "").replace("'s", " 's")
        )

        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, "wb"))
        print("dictionary dumped to %s" % path)

    @classmethod
    def load_from_file(cls, path):
        print("loading dictionary from %s" % path)
        word2idx, idx2word = cPickle.load(open(path, "rb"))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _get_img_id(image_name):
    return image_name.split(".")[0]


def _create_entry(img, annotation, answer):
    if answer:
        answer.pop("image")
        answer.pop("question_id")
    entry = {
        "question_id": annotation["question_id"],
        "image_id": _get_img_id(annotation["image"]),
        "image": img,
        "question": annotation["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    annotation_path = os.path.join(
        dataroot,
        "Annotations/%s.json" % name,
    )
    annotations = json.load(open(annotation_path))
    question_id = 0
    for annotation in annotations:
        annotation["question_id"] = "{}_{}".format(name, question_id)
        question_id += 1

    annotations = sorted(annotations, key=lambda x: x["question_id"])

    if "test" != name:  # train, val
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        answers = cPickle.load(open(answer_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

        assert_eq(len(annotations), len(answers))
        entries = []

        for annotation, answer in zip(annotations, answers):
            assert_eq(annotation["question_id"], answer["question_id"])
            assert_eq(annotation["image"], answer["image"])

            img_id = _get_img_id(annotation["image"])
            entry = _create_entry(img_id2val[img_id], annotation, answer)
            entries.append(entry)

    else:  # test
        entries = []
        for annotation in annotations:
            img_id = _get_img_id(annotation["image"])
            entry = _create_entry(img_id2val[img_id], annotation, None)
            entries.append(entry)
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(
        self,
        name,
        dictionary,
        dataroot="data",
    ):
        super(VQAFeatureDataset, self).__init__()
        assert name in ["train", "val", "test"]
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

        self.dictionary = dictionary

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

        self.entries = _load_dataset(
            dataroot, name, self.img_id2idx, self.label2ans
        )
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=MAX_QUES_SEQ_LEN):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry["question"], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (
                    max_length - len(tokens)
                )
                tokens = tokens + padding
            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            answer = entry["answer"]
            if answer:
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = torch.from_numpy(self.features[entry["image"]])

        question = entry["q_token"]
        image_id = entry["image_id"]
        answer = entry["answer"]
        if answer:
            labels = answer["labels"]
            scores = answer["scores"]
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return (features, torch.tensor([]), question, target)
        else:
            bb = torch.from_numpy(self.bboxes[entry["image"]])
            return (
                features,
                bb,
                question,
                image_id,
            )

    def __len__(self):
        return len(self.entries)


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)
