"""
Word2Vec Tiny Word Embedding Trainer (plotting throughout training) in 3D
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
vocab = ["apple", "mango", "orange", "fruit",  "broccoli",
         "potato", "spinach", "vegetable", "soft", "firm"]
pairs = [
    ("apple", "fruit"),
    ("mango", "fruit"),
    ("orange", "fruit"),
    ("broccoli", "vegetable"),
    ("potato", "vegetable"),
    ("spinach", "vegetable"),
    ("apple", "firm"),
    ("mango", "firm"),
    ("orange", "soft"),
    ("broccoli", "firm"),
    ("potato", "firm"),
    ("spinach", "soft"),
]

# --- Model parameters ---
V = len(vocab)   # vocabulary size
N = 3            # embedding dimensions (hidden layer size)
epochs = 300     # training cycles
lr = 0.05        # learning rate

one_hot = np.diag(np.ones(len(vocab)))  # one-hot vectors
W = np.random.uniform(-1, 1, (V, N))
U = np.random.uniform(-1, 1, (N, V))

# --- Enabling interactive plotting ---
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

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
        x_coord, y_coord, z_coord = W[i]
        ax.scatter(x_coord, y_coord, z_coord, color='green' if i >
                   3 else 'red', marker='o' if i in [1, 2, 6, 9] else '+')
        ax.text(x_coord + 0.1, y_coord + 0.1,  z_coord + 0.1, word, fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title(f"Embeddings after {epoch+1} epochs")
    ax.grid(True)
    plt.pause(0.01)

plt.ioff()
plt.show()
