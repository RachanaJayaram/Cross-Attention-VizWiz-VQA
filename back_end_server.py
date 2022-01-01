"""Backend Server for Demo. 
Loads pretrained model for inference."""
import os
import sys

from absl import app as application
from flask import Flask, json, jsonify, make_response, request
from flask_classful import FlaskView, route

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.predict import Inference
from demo.visualize import (
    visualize_question_attention,
    visualize_image_attention,
)

HOST = "0.0.0.0"
PORT = "3000"
app = Flask(__name__)


class ServerView(FlaskView):
    def __init__(self):
        self._inference_object = Inference()

    def index(self):
        return "Vizwiz"

    @route("/GetAnswerVizwiz", methods=["GET", "POST"])
    def get_answer_vizwiz(self):
        question = request.form["question"]
        image_id = int(request.form["image_id"])
        print(image_id, question)
        (answer, i_att, q_att, bboxes) = self._inference_object.get_prediction(
            image_id, question
        )

        visualize_question_attention(question, image_id, q_att)
        visualize_image_attention(
            image_id, bboxes[0], torch.flatten(i_att).cpu().detach().numpy()
        )
        try:
            return make_response(
                jsonify(
                    {
                        "answer": answer,
                    }
                )
            )
        except Exception as error:
            return make_response(jsonify({"Vizwiz": "Error {}".format(error)}))


def main(_):
    ServerView.register(app)
    app.run(HOST, PORT, debug=True)


if __name__ == "__main__":
    application.run(main)
