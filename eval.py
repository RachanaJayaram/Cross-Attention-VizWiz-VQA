import json
import os

import torch
import torch.nn as nn
from absl import app
from torch.utils.data import DataLoader

import utils.dataset as dataset
import utils.train_utils as train_utils
from train import evaluate
from utils.flags import FLAGS
from model.vqa_model import VQAModel, ModelParams


def main(_):
    """Main function."""
    train_utils.create_dir(FLAGS.save_folder)
    logger = train_utils.get_logger("VQA", FLAGS.save_folder)

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True

    data_params = json.load(open(FLAGS.data_params_path))
    dictionary = dataset.Dictionary.load_from_file(FLAGS.dictionary_path)

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
    model = nn.DataParallel(model).cuda()
    model.train(False)

    eval_dset = dataset.VQAFeatureDataset("val", dictionary)
    eval_loader = DataLoader(
        eval_dset, FLAGS.batch_size, shuffle=True, num_workers=1
    )
    if not FLAGS.snapshot_path:
        paths = [
            os.path.join(FLAGS.save_folder, file_path)
            for file_path in os.listdir(FLAGS.save_folder)
            if os.path.isfile(os.path.join(FLAGS.save_folder, file_path))
        ]
    else:
        paths = [FLAGS.snapshot_path]
    for path in paths:
        model_data = torch.load(path)
        model.load_state_dict(model_data.get("model_state", model_data))

        eval_score, bound = evaluate(model, eval_loader)
        logger.info(
            "epoch %d eval score: %.2f,\t" "train score: %.2f,\t" "bound: %.2f",
            model_data.get("epoch", model_data),
            100 * eval_score,
            model_data.get("score", model_data),
            100 * bound,
        )


if __name__ == "__main__":
    app.run(main)
