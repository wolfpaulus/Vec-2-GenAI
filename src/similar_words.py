""" Find most similar words to a given input word using preprocessed GloVe embeddings """
from sys import argv
import numpy as np


def load_vectors(filepath: str) -> dict[str, np.ndarray]:
    """
    Load GloVe vectors from a file into a dictionary
    :param filepath: Path to the GloVe file (preprocessed and normalized)
    :return: Dictionary mapping words to their vector representations
    """
    vectors = {}
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return vectors


def most_similar(vectors: dict[str, np.ndarray], target_word: str, top_k: int = 10) -> list[tuple[float, str]]:
    """
    Find the top_k most similar words to the target_word based on cosine similarity
    :param vectors: Dictionary of word vectors
    :param target_word: The word to find similar words for
    :param top_k: Number of similar words to return
    :return: List of tuples (similarity, word) sorted by similarity in descending order
    """
    similarities = []
    target = vectors.get(target_word)
    if target is not None:
        for word, vec in vectors.items():
            if word == target_word:
                continue
            sim = np.dot(target, vec)  # cosine similarity since vectors are normalized
            if len(similarities) < top_k:
                similarities.append((sim, word))
                similarities.sort(reverse=True, key=lambda x: x[0])
            elif sim > similarities[-1][0]:
                similarities[-1] = (sim, word)
                similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities


if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python src/similar_words.py <glove_filepath>")
        exit(1)
    vectors = load_vectors(argv[1])
    while True:
        target_word = input("Enter a word to find similar words: ").strip().lower()
        neighbors = most_similar(vectors, target_word)
        print(f"\nTop 10 words similar to '{target_word}':")
        for sim, w in neighbors:
            print(f"{w:15}  {sim:.3f}")
