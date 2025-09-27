""" Find most similar words to a given input word using preprocessed GloVe embeddings """
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


def most_similar(v: dict[str, np.ndarray], word: str, top_k: int = 10) -> list[tuple[float, str]]:
    """
    Find the top_k most similar words to the word based on cosine similarity
    :param v: Dictionary of word vectors
    :param word: The word to find similar words for
    :param top_k: Number of similar words to return
    :return: List of tuples (similarity, word) sorted by similarity in descending order
    """
    sims = []
    target = v.get(word)
    if target is not None:
        for w, vec in v.items():
            if w != word:
                similarity = np.dot(target, vec)  # cosine similarity for normalized vectors
                if len(sims) < top_k or similarity > sims[-1][0]:
                    sims.append((similarity, w))
                    sims.sort(reverse=True, key=lambda x: x[0])
    return sims


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python src/similar_words.py <glove_filepath>")
        exit(1)
    vectors = load_vectors(argv[1])
    while True:
        target_word = input("Enter a word to find similar words: ").strip().lower()
        neighbors = most_similar(vectors, target_word)
        print(f"\nTop 10 words similar to '{target_word}':")
        for sim, wrd in neighbors:
            print(f"{wrd:15}  {sim:.3f}")
