"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import os
import sys
import json
import numpy as np
import re
import pickle
import errno


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn’t’ve",
    "couldnt’ve": "couldn’t’ve",
    "didnt": "didn’t",
    "doesnt": "doesn’t",
    "dont": "don’t",
    "hadnt": "hadn’t",
    "hadnt’ve": "hadn’t’ve",
    "hadn'tve": "hadn’t’ve",
    "hasnt": "hasn’t",
    "havent": "haven’t",
    "hed": "he’d",
    "hed’ve": "he’d’ve",
    "he’dve": "he’d’ve",
    "hes": "he’s",
    "howd": "how’d",
    "howll": "how’ll",
    "hows": "how’s",
    "Id’ve": "I’d’ve",
    "I’dve": "I’d’ve",
    "Im": "I’m",
    "Ive": "I’ve",
    "isnt": "isn’t",
    "itd": "it’d",
    "itd’ve": "it’d’ve",
    "it’dve": "it’d’ve",
    "itll": "it’ll",
    "let’s": "let’s",
    "maam": "ma’am",
    "mightnt": "mightn’t",
    "mightnt’ve": "mightn’t’ve",
    "mightn’tve": "mightn’t’ve",
    "mightve": "might’ve",
    "mustnt": "mustn’t",
    "mustve": "must’ve",
    "neednt": "needn’t",
    "notve": "not’ve",
    "oclock": "o’clock",
    "oughtnt": "oughtn’t",
    "ow’s’at": "’ow’s’at",
    "’ows’at": "’ow’s’at",
    "’ow’sat": "’ow’s’at",
    "shant": "shan’t",
    "shed’ve": "she’d’ve",
    "she’dve": "she’d’ve",
    "she’s": "she’s",
    "shouldve": "should’ve",
    "shouldnt": "shouldn’t",
    "shouldnt’ve": "shouldn’t’ve",
    "shouldn’tve": "shouldn’t’ve",
    "somebody’d": "somebodyd",
    "somebodyd’ve": "somebody’d’ve",
    "somebody’dve": "somebody’d’ve",
    "somebodyll": "somebody’ll",
    "somebodys": "somebody’s",
    "someoned": "someone’d",
    "someoned’ve": "someone’d’ve",
    "someone’dve": "someone’d’ve",
    "someonell": "someone’ll",
    "someones": "someone’s",
    "somethingd": "something’d",
    "somethingd’ve": "something’d’ve",
    "something’dve": "something’d’ve",
    "somethingll": "something’ll",
    "thats": "that’s",
    "thered": "there’d",
    "thered’ve": "there’d’ve",
    "there’dve": "there’d’ve",
    "therere": "there’re",
    "theres": "there’s",
    "theyd": "they’d",
    "theyd’ve": "they’d’ve",
    "they’dve": "they’d’ve",
    "theyll": "they’ll",
    "theyre": "they’re",
    "theyve": "they’ve",
    "twas": "’twas",
    "wasnt": "wasn’t",
    "wed’ve": "we’d’ve",
    "we’dve": "we’d’ve",
    "weve": "we've",
    "werent": "weren’t",
    "whatll": "what’ll",
    "whatre": "what’re",
    "whats": "what’s",
    "whatve": "what’ve",
    "whens": "when’s",
    "whered": "where’d",
    "wheres": "where's",
    "whereve": "where’ve",
    "whod": "who’d",
    "whod’ve": "who’d’ve",
    "who’dve": "who’d’ve",
    "wholl": "who’ll",
    "whos": "who’s",
    "whove": "who've",
    "whyll": "why’ll",
    "whyre": "why’re",
    "whys": "why’s",
    "wont": "won’t",
    "wouldve": "would’ve",
    "wouldnt": "wouldn’t",
    "wouldnt’ve": "wouldn’t’ve",
    "wouldn’tve": "wouldn’t’ve",
    "yall": "y’all",
    "yall’ll": "y’all’ll",
    "y’allll": "y’all’ll",
    "yall’d’ve": "y’all’d’ve",
    "y’alld’ve": "y’all’d’ve",
    "y’all’dve": "y’all’d’ve",
    "youd": "you’d",
    "youd’ve": "you’d’ve",
    "you’dve": "you’d’ve",
    "youll": "you’ll",
    "youre": "you’re",
    "youve": "you’ve",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


# Notice that VQA score is the average of 10 choose 9 candidate answers cases
# See http://visualqa.org/evaluation.html
def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(comma_strip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version"""
    occurence = {}

    for ans_entry in answers_dset:
        answers = ans_entry["answers"]
        for answer in answers:
            answer = preprocess_answer(answer["answer"])
            if answer not in occurence:
                occurence[answer] = set()
            occurence[answer].add(ans_entry["question_id"])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print(
        "Num of answers that appear >= %d times: %d"
        % (min_occurence, len(occurence))
    )
    return occurence


def create_ans2label(occurence, name, cache_root="data/cache"):
    """Note that this will also create label2ans.pkl at the same time
    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    create_dir(cache_root)

    cache_file = os.path.join(cache_root, name + "_ans2label.pkl")
    pickle.dump(ans2label, open(cache_file, "wb"))
    cache_file = os.path.join(cache_root, name + "_label2ans.pkl")
    pickle.dump(label2ans, open(cache_file, "wb"))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root="data/cache"):
    """Augment answers_dset with soft score as label
    ***answers_dset should be preprocessed***
    Write result into a cache file
    """
    target = []
    for ans_entry in answers_dset:
        answer_count = {}
        for answer in ans_entry["answers"]:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append(
            {
                "question_id": ans_entry["question_id"],
                "image": ans_entry["image"],
                "labels": labels,
                "scores": scores,
            }
        )

    create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + "_target.pkl")
    pickle.dump(target, open(cache_file, "wb"))
    return target


def get_answer(qid, answers):
    for ans in answers:
        if ans["question_id"] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question["question_id"] == qid:
            return question


def get_answers_dset(annotation_file, split):
    answers_dset = []
    question_id = 0
    for entry in json.load(open(annotation_file)):
        answers_dset.extend(
            [
                {
                    "answers": entry["answers"],
                    "image": entry["image"],
                    "question_id": "{}_{}".format(split, question_id),
                }
            ]
        )
        question_id += 1
    return answers_dset


if __name__ == "__main__":
    TRAIN_FILE = "data/Annotations/train.json"
    train_answers_dset = get_answers_dset(TRAIN_FILE, "train")

    VAL_FILE = "data/Annotations/val.json"
    val_answers_dset = get_answers_dset(VAL_FILE, "val")

    answers = train_answers_dset + val_answers_dset
    occurence = filter_answers(answers, 9)

    CACHE_PATH = "data/cache/trainval_ans2label.pkl"
    if os.path.isfile(CACHE_PATH):
        print("found %s" % CACHE_PATH)
        ans2label = pickle.load(open(CACHE_PATH, "rb"))
    else:
        ans2label = create_ans2label(occurence, "trainval")
    compute_target(train_answers_dset, ans2label, "train")
    compute_target(val_answers_dset, ans2label, "val")
