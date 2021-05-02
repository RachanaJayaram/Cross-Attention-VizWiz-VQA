"""Module containing functions to aide multimodal fusion."""


def calculate_similarity_matrix(v_proj, q_proj):
    """Calculates similarity matrix.

    The similarity matrix M is computed based on the distance between the input
    features of the question and image. The distance between the i-th object in
    the image and j-th word is calculated as the matrix multiplication of the
    object feature vector and the word feature vector.

    Args:
        v_proj:
            Visual feature tensor after projection.
            Dimensions are (batch_size, number_of_objects, hidden_dimension).
        q_proj:
            Question feature tensor after projection.
            Dimensions are (batch_size, number_of_words, hidden_dimension).

    Returns: A similarity matrix which is a 2d tensor of size
        (number_of_objects, number_of_words).
    """
    q_proj = q_proj.transpose(1, 2)
    similarity_matrix = v_proj @ q_proj
    return similarity_matrix
