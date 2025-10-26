""" Find the missing word in an analogy using preprocessed GloVe embeddings. """
import numpy as np
from mdl_utils import load_vectors


def analogy(words: list[str], v: dict[str, np.ndarray]) -> str:
    """
    Finds the missing word in an analogy:
    :param words: List of three words [word0, word1, word2]
    :param v: Dictionary of word vectors
    :return: The word that best completes the analogy word0 is to word1 as word2 is to ?
    """
    target = v[words[2]] - v[words[0]] + v[words[1]]  # vector arithmetic
    target /= np.linalg.norm(target)   # normalize the resulting vector
    # Find the closest word to the target vector
    return max(v.keys(), key=lambda w: np.dot(v[w], target) if w not in words else float('-inf'))


if __name__ == "__main__":
    vectors = load_vectors("models/wikigiga_50d.txt")
    while True:
        wrds = input("Enter 3 words like 'man woman king': ").strip().lower().split()
        if len(wrds) == 0:
            break
        if len(wrds) != 3:
            print("Please enter exactly 3 words.")
            continue
        try:
            print(f"{wrds[0]} is to {wrds[1]} as {wrds[2]} is to {analogy(wrds, vectors)}\n")
        except KeyError as e:
            print(f"'{e}' was not found in the vocabulary")
