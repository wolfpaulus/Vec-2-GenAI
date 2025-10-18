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


def most_similar(word: str, v: dict[str, np.ndarray], top_k: int = 10) -> list[tuple[str, float]]:
    """
    Find the top_k most similar words to the given word based on cosine similarity
    :param word: The word to find similar words for
    :param v: Dictionary of word vectors
    :param top_k: Number of similar words to return
    :return: List of tuples (word, similarity) sorted by similarity in descending order
    """
    sims = []
    target = v.get(word)
    if target is not None:
        for w, vec in v.items():  # Iterate over all word vectors (those are normalized already)
            if w != word:
                similarity = np.dot(target, vec)  # compare with the target word
                if len(sims) < top_k or similarity > sims[-1][1]:  # only keep top_k
                    sims.append((w, similarity))
                    sims.sort(reverse=True, key=lambda x: x[1])  # keep sorted
                    if len(sims) > top_k:
                        sims.pop()
    return sims


if __name__ == "__main__":
    mdl = "models/wikigiga_50d.txt" if len(argv) != 2 else argv[1]
    vectors = load_vectors(mdl)
    print(f"Loaded {len(vectors):,} word vectors from {mdl}")
    while True:
        target_word = input("Enter a word to find similar words: ").strip().lower()
        if target_word == "":
            break
        neighbors = most_similar(target_word, vectors)
        print(f"\nTop {len(neighbors)} words similar to '{target_word}':")
        for wrd, sim in neighbors:
            print(f"{wrd:15}  {sim:.3f}")
