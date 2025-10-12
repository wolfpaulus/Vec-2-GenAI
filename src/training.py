""" Tiny Word Embedding Trainer (inputs-only update) """
import numpy as np

# --- Vocabulary and Corpus ---
vocab = ["apple", "mango", "orange", "fruit", "broccoli", "potato", "spinach", "vegetable"]
pairs = [
    ("apple", "fruit"),
    ("mango", "fruit"),
    ("orange", "fruit"),
    ("broccoli", "vegetable"),
    ("potato", "vegetable"),
    ("spinach", "vegetable")
]

# --- Model parameters ---
V = len(vocab)   # vocabulary size
N = 2            # embedding dimensions
epochs = 20      # training epochs
lr = 0.05        # learning rate

# --- Initialize embeddings randomly ---
one_hot = np.diag(np.ones(V))  # one-hot vectors
W = np.random.uniform(-1, 1, (V, N))

# --- Training Loop ---
for epoch in range(epochs):
    for w1, w2 in pairs:
        i, j = vocab.index(w1), vocab.index(w2)
        h = W[i]                            # embedding of input word: matmul(one_hot[i],W)
        z = np.matmul(h, W.T)               # raw scores for all vocab words
        error = z - one_hot[j]              # simple prediction error
        W[i] -= lr * np.matmul(W.T, error)  # update only input embedding

print(W)  # final embeddings
