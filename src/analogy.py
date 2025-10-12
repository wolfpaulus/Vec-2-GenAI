"""
Find the missing word in an analogy using preprocessed GloVe embeddings.
E.g., KING is to QUEEN as MAN is to WOMAN
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


def analogy(words: list[str], v: dict[str, np.ndarray]) -> str:
    """
    Finds the missing word in an analogy:
    :param words: List of three words [word1, word2, word3]
    :param vectors: Dictionary of word vectors
    :return: The word that best completes the analogy 1 is to 2 as 3 is to ?
    """
    target = v[words[2]] - v[words[0]] + v[words[1]]  # e.g.: KING - QUEEN == MAN - WOMAN
    target /= np.linalg.norm(target)   # normalize the resulting vector
    # Find the closest word to the target vector
    return max(v.keys(), key=lambda w: np.dot(v[w], target) if w not in words else float('-inf'))


if __name__ == "__main__":
    mdl = "models/wikigiga_50d.txt" if len(argv) != 2 else argv[1]
    vectors = load_vectors(mdl)
    print(f"Loaded {len(vectors):,} word vectors from {mdl}")
    while True:
        words = input("Enter 3 words like 'king queen man': ").strip().lower().split()
        if len(words) != 3:
            print("Please enter exactly 3 words.")
            continue
        try:
            print(f"{words[0]} is to {words[1]} as {words[2]} is to {analogy(words, vectors)}\n")
        except KeyError as e:
            print(f"'{e}' was not found in the vocabulary")
