import copy
import json

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def attention_bbox_interpolation(im, bboxes, att):
    cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
    cmap.set_bad(color="k", alpha=0.0)
    softmax = att
    assert len(softmax) == len(bboxes)

    img_h, img_w = im.shape[:2]
    opacity = np.zeros((img_h, img_w), np.float32)
    for bbox, weight in zip(bboxes, softmax):
        x_1, y_1, x_2, y_2 = bbox.cpu().numpy()
        opacity[int(y_1) : int(y_2), int(x_1) : int(x_2)] += weight
    opacity = np.minimum(opacity, 1)

    opacity = opacity[..., np.newaxis]

    vis_im = np.array(Image.fromarray(cmap(opacity, bytes=True), "RGBA"))
    vis_im = vis_im.astype(im.dtype)
    vis_im = cv2.addWeighted(im, 0.7, vis_im, 0.5, 0)
    vis_im = vis_im.astype(im.dtype)

    return vis_im


def visualize_image_attention(image_id, boxes, att_weights):
    im = cv2.imread(
        "/home/rachana/Documents/demo/static/img/{}.jpg".format(image_id)
    )
    print(boxes)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    b, g, r, a = cv2.split(im)  # get b, g, r
    im = cv2.merge([r, g, b, a])

    M = min(len(boxes), len(att_weights))
    im_ocr_att = attention_bbox_interpolation(im, boxes[:M], att_weights[:M])
    plt.imsave(
        "/home/rachana/Documents/demo/static/attention_maps/{}.png".format(
            image_id
        ),
        im_ocr_att,
    )


def visualize_question_attention(question, image_id, q_att):

    sentence = question.lower()
    sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's")
    words = sentence.split()[:14]
    q_att = torch.flatten(q_att).cpu().detach()[: len(words)]
    q_att -= q_att.min(0, keepdim=True)[0]
    q_att /= q_att.max(0, keepdim=True)[0]
    q_att = q_att.numpy()
    att_list = [
        {"word": words[i], "attention": q_att[i].item()}
        for i in range(len(words))
    ]
    with open(
        "/home/rachana/Documents/demo/static/question_maps/{}.json".format(
            image_id
        ),
        "w",
    ) as outfile:
        outfile.write(json.dumps(att_list))
