import attr
import torch
import torch.nn as nn

from model.attention import Attention, ReAttention
from model.classification import Classifier
from model.fusion import calculate_similarity_matrix
from model.multi_layer_net import MultiLayerNet
from model.question_embedding import QuestionEmbedding, WordEmbedding
from model.graph.relation_encoder import RelationEncoder


@attr.s
class ModelParams:
    """Stores configurations specific to training.

    Attributes:
        add_reattention(bool): Reattention will be performed if set to true.
        add_graph_attention(bool): Graph attention will be performed if set to
            true.
        question_sequence_length(int): Number of tokens in the question.
        number_of_objects (int): Number of objects in the image embedding.
        word_embedding_dimension (int): Dimension of word embedding.
        object_embedding_dimension (int): Dimension of object visual feature.
        vocabulary_size (int): Size of the vocabulary of the word embedder.
        num_ans_candidates (int): Number of answer candiadates considered.
    """

    add_reattention: bool = attr.ib()
    add_graph_attention: bool = attr.ib()
    question_sequence_length: int = attr.ib()
    number_of_objects: int = attr.ib()
    word_embedding_dimension: int = attr.ib()
    object_embedding_dimension: int = attr.ib()
    vocabulary_size: int = attr.ib()
    attention_heads: int = attr.ib()
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
        )
        self.question_embedding_net = QuestionEmbedding(
            input_dimension=model_params.word_embedding_dimension,
            number_hidden_units=hidden_dimension,
            number_of_layers=1,
        )

        self.relation_net = None
        if model_params.add_graph_attention:
            self.relation_net = RelationEncoder(
                model_params.object_embedding_dimension,
                hidden_dimension,
                hidden_dimension,
                model_params.attention_heads,
            )

        self.question_projection_net = MultiLayerNet(
            dimensions=[hidden_dimension, hidden_dimension], dropout=0.2
        )
        self.image_projection_net = MultiLayerNet(
            dimensions=[
                model_params.object_embedding_dimension,
                hidden_dimension,
            ],
            dropout=0.2,
        )

        self.question_attention_net = Attention(model_params.number_of_objects)
        self.visual_attention_net = Attention(
            model_params.question_sequence_length
        )

        self.answer_projection_net = MultiLayerNet(
            dimensions=[hidden_dimension * 2, hidden_dimension], dropout=0.2
        )
        self.classifier = Classifier(
            input_dimension=hidden_dimension,
            hidden_dimension=2 * hidden_dimension,
            output_dimension=model_params.num_ans_candidates,
            dropout=0.5,
        )

        self.reattention_net = None
        if model_params.add_reattention:
            self.reattention_net = ReAttention(
                hidden_dimension, model_params.number_of_objects
            )

    def reattention_added(self):
        """Returns boolean value indicating whether reattention was added."""
        return self.model_params.add_reattention

    def _get_attented_features(
        self, similarity_matrix, attention_layer, feature_vector
    ):
        attention_weights = attention_layer(similarity_matrix)
        attended_vector = (attention_weights * feature_vector).sum(1)
        return attended_vector, attention_weights

    def forward(self, v, q, spa_adj_matrix=None):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        word_embedding = self.word_embedding_net(q)
        question_embedding = self.question_embedding_net(word_embedding)
        proj_question_embedding = self.question_projection_net(
            question_embedding
        )

        if self.model_params.add_graph_attention:
            proj_image_embedding = self.relation_net(
                v, proj_question_embedding, spa_adj_matrix
            )
        else:
            proj_image_embedding = self.image_projection_net(v)

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
            similarity_matrix, self.visual_attention_net, proj_image_embedding
        )
        attended_image_embedding, visual_attention_weights = att_return_values

        joint_representation = torch.cat(
            (attended_question_embedding, attended_image_embedding), dim=1
        )
        proj_joint_representation = self.answer_projection_net(
            joint_representation
        )
        logits = self.classifier(proj_joint_representation)

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
