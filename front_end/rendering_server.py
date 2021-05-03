import os
import socket
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import requests
from flask import (
    Flask,
    Response,
    json,
    jsonify,
    make_response,
    render_template,
    request,
    send_file,
    url_for,
)

app = Flask(__name__)
HOST = "127.0.0.1"
PORT = "4200"
URL_VIZWIZ = "http://0.0.0.0:3000/server/GetAnswerVizwiz"
URL_VQA = "http://0.0.0.0:4000/server/GetAnswerVQA"


@app.route("/")
def default():
    return "<h1>Visual Question Answering</h1>"


@app.route("/VqaDemo")
def VqaDemo():
    return render_template("vqa.html", server_url="{}:{}".format(HOST, PORT))


@app.route("/VqaQuery", methods=["GET", "POST"])
def VqaQuery():
    if request.method == "POST":
        img_id = request.form["img_id"]

        return render_template(
            "vqa.html",
            img_id=img_id,
            img_url="{}/img/{}.jpg".format("static", img_id),
        )


@app.route("/getAnswer", methods=["GET", "POST"])
def getAnswer():
    if request.method == "POST":
        backend_request_data = {
            "image_id": request.form["img_id"],
            "question": request.form["question"],
        }
        model_answers = []
        try:
            response_vizwiz = requests.post(
                URL_VIZWIZ, data=backend_request_data
            ).json()
            model_answers.append(
                {
                    "name": "Vizwiz",
                    "acc": "52%",
                    "answer": response_vizwiz["answer"],
                }
            )
        except:
            pass
        try:
            response_vqa = requests.post(
                URL_VQA, data=backend_request_data
            ).json()
            model_answers.append(
                {
                    "name": "VQA",
                    "acc": "63%",
                    "answer": response_vqa["answer"],
                }
            )
        except:
            pass
        attention_map_path = "{}/{}/{}.png".format(
            "static", "attention_maps", request.form["img_id"]
        )
        question_map = json.load(
            open("static/question_maps/{}.json".format(request.form["img_id"]))
        )

        data = {
            "img_id": request.form["img_id"],
            "question": request.form["question"],
            "att_map_url": attention_map_path,
            "question_map": question_map,
            "model_answers": model_answers,
        }
        return make_response(jsonify(data), 201)


if __name__ == "__main__":
    app.run(HOST, PORT, debug=True)
