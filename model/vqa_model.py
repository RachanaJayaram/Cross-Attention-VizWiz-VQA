import attr
import torch
import torch.nn as nn

from model.attention import Attention, ReAttention, SelfAttention
from model.classification import Classifier
from model.fusion import calculate_similarity_matrix
from model.multi_layer_net import MultiLayerNet
from model.question_embedding import QuestionEmbedding, WordEmbedding
from utils.flags import FusionMethod


@attr.s
class ModelParams:
    """Stores configurations specific to training.

    Attributes:
        add_reattention(bool): Reattention will be performed if set to true.
        fusion_method(FusionMethod): Determines how the joint representation is
            obtained.
        question_sequence_length(int): Number of tokens in the question.
        number_of_objects (int): Number of objects in the image embedding.
        word_embedding_dimension (int): Dimension of word embedding.
        object_embedding_dimension (int): Dimension of object visual feature.
        vocabulary_size (int): Size of the vocabulary of the word embedder.
        num_ans_candidates (int): Number of answer candiadates considered.
    """

    add_self_attention: bool = attr.ib()
    add_reattention: bool = attr.ib()
    fusion_method: FusionMethod = attr.ib()
    question_sequence_length: int = attr.ib()
    number_of_objects: int = attr.ib()
    word_embedding_dimension: int = attr.ib()
    object_embedding_dimension: int = attr.ib()
    vocabulary_size: int = attr.ib()
    num_ans_candidates: int = attr.ib()


class VQAModel(nn.Module):
    def __init__(
        self,
        glove_path: str,
        model_params: ModelParams,
        hidden_dimension: int,
    ):
        super().__init__()
        self.model_params = model_params
        self.word_embedding_net = WordEmbedding(
            vocabulary_size=model_params.vocabulary_size,
            pretrained_vectors_file=glove_path,
            embedding_dimension=model_params.word_embedding_dimension,
            dropout=0.25,
        )
        self.question_embedding_net = QuestionEmbedding(
            input_dimension=model_params.word_embedding_dimension,
            number_hidden_units=hidden_dimension,
            number_of_layers=1,
        )

        self.question_projection_net = MultiLayerNet(
            dimensions=[hidden_dimension, hidden_dimension], dropout=0.5
        )
        self.image_projection_net = MultiLayerNet(
            dimensions=[
                model_params.object_embedding_dimension,
                hidden_dimension,
            ],
            dropout=0.5,
        )

        if self.model_params.add_self_attention:
            self.question_self_attention_net = SelfAttention(
                hidden_dimension, dropout=0.3
            )
            self.visual_self_attention_net = SelfAttention(
                hidden_dimension, dropout=0.3
            )

        self.question_attention_net = Attention(
            model_params.number_of_objects, dropout=0.3
        )
        self.visual_attention_net = Attention(
            model_params.question_sequence_length, dropout=0.3
        )

        if model_params.fusion_method == FusionMethod.CONCAT:
            factor = 3 if self.model_params.add_self_attention else 2
            self.classifier = Classifier(
                input_dimension=hidden_dimension * 3,
                hidden_dimension=hidden_dimension * 4,
                output_dimension=model_params.num_ans_candidates,
                dropout=0.5,
            )
        elif model_params.fusion_method == FusionMethod.HADAMARD:
            self.classifier = Classifier(
                input_dimension=hidden_dimension,
                hidden_dimension=hidden_dimension * 2,
                output_dimension=model_params.num_ans_candidates,
                dropout=0.5,
            )

        self.reattention_net = None
        if model_params.add_reattention:
            self.reattention_net = ReAttention(
                hidden_dimension,
                model_params.number_of_objects,
                self.model_params.fusion_method,
                0.3,
            )

    def reattention_added(self):
        """Returns boolean value indicating whether reattention was added."""
        return self.model_params.add_reattention

    def _get_attented_features(
        self, attention_input, attention_layer, feature_vector
    ):
        attention_weights = attention_layer(attention_input)
        attended_vector = (attention_weights * feature_vector).sum(1)
        return attended_vector, attention_weights

    def forward(self, v, q):
        """Forward."""
        word_embedding = self.word_embedding_net(q)
        question_embedding = self.question_embedding_net(word_embedding)
        proj_question_embedding = self.question_projection_net(
            question_embedding
        )

        proj_image_embedding = self.image_projection_net(v)

        if self.model_params.add_self_attention:
            (
                self_attended_image_embedding,
                visual_attention_weights,
            ) = self._get_attented_features(
                proj_image_embedding,
                self.visual_self_attention_net,
                proj_image_embedding,
            )

            (
                self_attended_question_embedding,
                question_attention_weights,
            ) = self._get_attented_features(
                proj_question_embedding,
                self.question_self_attention_net,
                proj_question_embedding,
            )

        similarity_matrix = calculate_similarity_matrix(
            proj_image_embedding, proj_question_embedding
        )

        (
            attended_question_embedding,
            question_attention_weights,
        ) = self._get_attented_features(
            similarity_matrix.transpose(1, 2),
            self.question_attention_net,
            proj_question_embedding,
        )
        att_return_values = self._get_attented_features(
            similarity_matrix,
            self.visual_attention_net,
            proj_image_embedding,
        )
        (
            attended_image_embedding,
            visual_attention_weights,
        ) = att_return_values

        if self.model_params.fusion_method == FusionMethod.CONCAT:
            joint_representation = (
                torch.cat(
                    (
                        # self_attended_image_embedding,
                        attended_image_embedding,
                        self_attended_question_embedding,
                        attended_question_embedding,
                    ),
                    dim=1,
                )
                if self.model_params.add_self_attention
                else torch.cat(
                    (
                        attended_image_embedding,
                        attended_question_embedding,
                    ),
                    dim=1,
                )
            )

        elif self.model_params.fusion_method == FusionMethod.HADAMARD:
            joint_representation = (
                attended_image_embedding * attended_question_embedding
            )
        logits = self.classifier(joint_representation)

        visual_re_attention_weights = None
        if self.model_params.add_reattention:
            visual_re_attention_weights = self.reattention_net(
                joint_representation,
                proj_question_embedding,
                proj_image_embedding,
            )
        return (
            logits,
            visual_attention_weights,
            visual_re_attention_weights,
            question_attention_weights,
        )
