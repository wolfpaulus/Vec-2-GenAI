"""
Finding the outlier in a list of words using preprocessed GloVe embeddings.
"""
from sys import argv
import numpy as np


def load_vectors(filepath: str) -> dict[str, np.ndarray]:
    """
    Load GloVe vectors from a file into a dictionary
    :param filepath: Path to the GloVe file (preprocessed and normalized)
    :return: Dictionary mapping words to their vector representations
    """
    word_vectors = dict()
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            word_vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return word_vectors


def outlier(v: dict[str, np.ndarray], words: list[str]) -> str:
    """
    Finds the outlier word in a list of words:
    :param vectors: Dictionary of word vectors
    :param words: List of words (more than 3)
    :return: The word that is least similar to the others
    """
    # Compute the mean vector of all words
    mean_vector = np.mean([v[word] for word in words], axis=0)
    mean_vector /= np.linalg.norm(mean_vector)  # normalize the mean vector

    # Find the word with the smallest cosine similarity to the mean vector
    outlier_word = min(words, key=lambda w: np.dot(v[w], mean_vector))
    return outlier_word


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python src/outlier.py <glove_filepath>")
        exit(1)
    vectors = load_vectors(argv[1])
    while True:
        words = input(
            "Enter more than 3 words like 'berlin paris london munich': ").strip().lower().split()
        if len(words) <= 3:
            print("Please enter more than 3 words.")
            continue
        try:
            print(f"The outlier is: {outlier(vectors, words)}\n")
        except KeyError as e:
            print(f"'{e}' was not found in the vocabulary")
