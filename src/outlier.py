""" Finding the outlier in a list of words using preprocessed GloVe embeddings. """
import numpy as np
from mdl_utils import load_vectors


def outlier(words: list[str], v: dict[str, np.ndarray]) -> str:
    """
    Finds the outlier word in a list of words, assumes the input word vectors are normalized.
    :param words: List of words (more than 3)
    :param v: Dictionary of word vectors
    :return: The word that is least similar to the others
    """
    # Compute the mean vector of all words
    mean_vector = np.mean([v[word] for word in words], axis=0)
    # Find the word with the smallest cosine similarity to the mean vector
    outlier_word = min(words, key=lambda w: np.dot(v[w], mean_vector))
    return outlier_word


if __name__ == "__main__":
    vectors = load_vectors("models/wikigiga_50d.txt")
    while True:
        wrds = input(
            "Enter more than 3 words like 'car bike house plane': ").strip().lower().split()
        if len(wrds) == 0:
            break
        if len(wrds) <= 3:
            print("Please enter more than 3 words.")
            continue
        try:
            print(f"The outlier is: {outlier(wrds, vectors)}\n")
        except KeyError as e:
            print(f"'{e}' was not found in the vocabulary")
