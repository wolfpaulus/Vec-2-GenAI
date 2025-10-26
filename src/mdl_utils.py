""" Utility functions for the application. """
from sys import argv
import numpy as np


def load_vectors(filepath: str) -> dict[str, np.ndarray]:
    """
    Load GloVe vectors from a file into a dictionary
    :param filepath: Path to the GloVe file (preprocessed and normalized)
    :return: Dictionary mapping words to their vector representations
    """
    try:
        word_vectors = dict()
        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                parts = line.split()
                word_vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
        print(f"Loaded {len(word_vectors):,} word vectors from {filepath}")
        return word_vectors
    except OSError as e:
        print(f"Error loading vectors from {filepath}: {e}")
        exit(1)


def preprocess(in_filepath: str, out_filepath: str) -> None:
    """
    Preprocess GloVe embeddings to shorten vocabulary and normalize vectors
    1. Keep only words with alphabetic characters (no numbers, punctuation, etc.)
    2. Keep only words in ⅓ Million Most Frequent English Words on the Web
    3. Normalize each vector to unit length
    4. Save the processed embeddings to a new file
    :param in_filepath: Path to the input GloVe file
    :param out_filepath: Path to the output preprocessed file
    :return: None
    """
    # Load the set of frequent words
    frequent_words = set()
    with open("./models/unigram_freq.csv", "r", encoding="utf8") as f:
        for line in f:
            frequent_words.add(line.strip().split(",")[0].lower())

    with open(in_filepath, "r", encoding="utf8") as in_f:
        with open(out_filepath, "w", encoding="utf8") as out_f:
            for line in in_f:
                parts = line.split()
                if parts[0].isalpha() and parts[0] in frequent_words:  # keep only alphabetic words in frequent list
                    try:
                        vec = np.array(parts[1:], dtype=np.float32)
                        vec = vec / np.linalg.norm(vec)   # normalize vector
                        out_f.write(f"{parts[0]} {' '.join(map(str, vec))}\n")
                    except ValueError as e:
                        print(f"Error processing line: {parts[:5]}\n{e}")


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python src/pre_proc.py <in_filepath> <out_filepath>")
        exit(1)
    preprocess(argv[1], argv[2])
