"""
Tiny Word Embedding Trainer (inputs-only update) - Plotting throughout training
"""
import numpy as np
import matplotlib.pyplot as plt

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

# --- Hyperparameters ---
V = len(vocab)   # vocabulary size
N = 2            # embedding dimensions
epochs = 20      # training epochs
lr = 0.05        # learning rate

# --- Initialize embeddings randomly ---
one_hot = np.diag(np.ones(V))  # one-hot vectors
W = np.random.uniform(-1, 1, (V, N))

# --- Enabling interactive plotting ---
plt.ion()
_, ax = plt.subplots()

# --- Training Loop ---
for epoch in range(epochs):
    for w1, w2 in pairs:
        i, j = vocab.index(w1), vocab.index(w2)
        h = W[i]                            # embedding of input word: matmul(one_hot[i],W)
        z = np.matmul(h, W.T)               # raw scores for all vocab words
        error = z - one_hot[j]              # simple prediction error
        W[i] -= lr * np.matmul(W.T, error)  # update only input embedding
    # --- Visualization ---
    ax.clear()
    for i, word in enumerate(vocab):
        x_coord, y_coord = W[i]
        ax.scatter(x_coord, y_coord, color='green' if i > 3 else 'red')
        ax.text(x_coord + 0.02, y_coord + 0.02, word, fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Embeddings after {epoch+1} epochs")
    ax.grid(True)
    plt.pause(0.07)
print(W)  # final embeddings
plt.ioff()
plt.show()
