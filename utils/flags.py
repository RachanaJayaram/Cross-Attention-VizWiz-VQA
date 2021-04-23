"""Module for flag definitions."""

from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_folder",
    "save_folder/first",
    "Folder where trained model will be saved.",
)
flags.DEFINE_string(
    "dictionary_path",
    "data/glove/dictionary.pkl",
    "Path to dictionary.pkl",
)
flags.DEFINE_string(
    "glove_path",
    "data/glove/glove6b_init_300d.npy",
    "Path to GloVe np file.",
)
flags.DEFINE_string(
    "data_params_path",
    "data/data_parameters.json",
    "Path to data_parameters.json file.",
)
flags.DEFINE_string(
    "snapshot_path",
    None,
    "Path to model snapshot.",
)
flags.DEFINE_integer(
    "batch_size",
    256,
    "The number of training examples in one forward/backward pass.",
)
flags.DEFINE_float(
    "base_learning_rate",
    1e-3,
    "The default rate at which weights are updated during training.",
)
flags.DEFINE_bool(
    "add_reattention",
    False,
    "Determines whether reattention should be performed.",
)
flags.DEFINE_bool(
    "add_graph_attention",
    False,
    "Determines whether graph attention should be performed.",
)

flags.DEFINE_integer("seed", 1204, "Random seed.")
flags.DEFINE_integer("hidden_dimension", 1024, "Dimension of hidden states.")
flags.DEFINE_integer("number_of_epochs", 30, "Number of epochs for training.")
flags.DEFINE_integer("attention_heads", 1, "Number of attention heads.")

flags.DEFINE_integer(
    "start_epoch", 0, "Epoch at which training should start/restart."
)
flags.DEFINE_integer(
    "warmup_length", 4, "Number of epochs for the warmup stage."
)
flags.DEFINE_float(
    "warmup_factor", 0.5, "Factor by which learning rate is multiplied."
)
flags.DEFINE_float(
    "decay_factor", 0.25, "Decay factor at which the learning rate is reduced."
)
flags.DEFINE_integer(
    "decay_start", 25, "Epoch at which learning rate decay should start."
)
flags.DEFINE_integer(
    "decay_step", 2, "Helps determine epochs for which decay is skipped."
)
flags.DEFINE_float(
    "grad_clip", 0.25, "Max norm of the gradients for grad clipping."
)

flags.DEFINE_float(
    "save_score_threshold",
    10,
    "Threshold for model score after which the models are saved.",
)
flags.DEFINE_integer(
    "save_step",
    1,
    "Determines epochs (by modulo) for which the model is saved after the "
    "score threshold is reached.",
)
