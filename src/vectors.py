"""
2-Dimensional Distributed Representation of Words
"""
import numpy as np
import matplotlib.pyplot as plt

vocab = ["apple", "mango", "orange", "fruit", "broccoli", "potato", "spinach",  "vegetable"]
V = len(vocab)   # vocabulary size
N = 2            # embedding dimensions

# Create a one-hot encoded vectors
one_hot_encoded = np.diag(np.ones(V))
print(one_hot_encoded)

# Create a two dimensional uniformly distributed representation for each word in the
W = np.random.uniform(-1, 1, (V, N))
print(W)

# Plot the distributed representation
for i, word in enumerate(vocab):
    x, y = W[i]
    plt.scatter(x, y,  color='green' if i > 3 else 'red')
    plt.text(x + 0.02, y + 0.02, word)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.title("Distributed Representation of Words")
plt.show()
