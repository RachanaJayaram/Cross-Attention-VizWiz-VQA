"""
This code modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
GNU General Public License v3.0
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import Dictionary


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        "Annotations/test.json",
        "Annotations/val.json",
        "Annotations/train.json",
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        samples = json.load(open(question_path))
        for sample in samples:
            dictionary.tokenize(sample["question"], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, "r") as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(" ")) - 1
    print("embedding dim is %d" % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(" ")
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == "__main__":
    d = create_dictionary("data")
    d.dump_to_file("data/glove/dictionary.pkl")

    d = Dictionary.load_from_file("data/glove/dictionary.pkl")
    emb_dim = 300
    glove_file = "data/glove/glove.6B.%dd.txt" % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save("data/glove/glove6b_init_%dd.npy" % emb_dim, weights)
