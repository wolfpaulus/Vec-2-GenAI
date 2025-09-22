"""
Word2Vec Tiny Word Embedding Trainer (plotting throughout training)
"""
import numpy as np
import matplotlib.pyplot as plt


def softmax(z: np.ndarray) -> np.ndarray:
    """ Turn scores into probabilities
        - exponentiates them (makes everything positive),
        - divides by their total (so they add up to 1).
      """
    e_z = np.exp(z)
    return e_z / e_z.sum()


# --- Vocabulary and Corpus ---
vocab = ["apple", "mango", "orange", "fruit",  "broccoli", "potato", "spinach", "vegetable"]
one_hot = np.diag(np.ones(len(vocab)))  # one-hot vectors
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
N = 2            # embedding dimensions (hidden layer size)
epochs = 200     # training cycles
lr = 0.05        # learning rate

W = np.random.uniform(-1, 1, (V, N))
U = np.random.uniform(-1, 1, (N, V))

# --- Enabling interactive plotting ---
plt.ion()
_, ax = plt.subplots()

# --- Training ---
for epoch in range(epochs):
    for w1, w2 in pairs:
        i, j = vocab.index(w1), vocab.index(w2)

        # --- Forward pass ---
        h = W[i]               # look up the embedding for our input word
        z = U.T @ h            # predict possible context words
        y = softmax(z)         # apply softmax

        # --- Backpropagation ---
        y_expected = one_hot[j]  # expected context one-hot (size V)
        error = y - y_expected   # prediction error (size V)

        # --- Update ---
        U -= lr * np.outer(h, error)    # update context matrix
        W[i] -= lr * (U @ error)        # update input word embedding

    # --- Visualization ---
    ax.clear()
    for i, word in enumerate(vocab):
        x_coord, y_coord = W[i]
        ax.scatter(x_coord, y_coord, color='green' if i > 3 else 'red')
        ax.text(x_coord + 0.1, y_coord + 0.1, word, fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(f"Embeddings after {epoch+1} epochs")
    ax.grid(True)
    plt.pause(0.02)
plt.ioff()
plt.show()
