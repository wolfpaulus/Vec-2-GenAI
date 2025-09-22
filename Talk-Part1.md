
# Title

- From Numbers to Meaning: The Building Blocks of Generative AI
.. or ..
- From Vectors to ChatGPT

## Objectives

- Understand how words and sentences can be represented as vectors.
- Learn how vectors capture meaning and relationships in language.
- See how neural networks use embeddings to generate responses.

---

## Generative AI

At its core: given an input sequence of words, the task is to guess the word most likely to follow.

This can be repeated again and again:

- Input: "Not all heroes wear"
  → Prediction: capes
- Input: "Not all heroes wear capes"
  → Prediction: but
- Input: "Not all heroes wear capes but"
  → Prediction: all
- … and so on until:
  **Not all heroes wear capes but all villains do.**

This is how generative AI builds up sentences — one word at a time.

---

## The Big Idea

For a computer, a word is just a sequence of characters — no meaning attached.

To fix that, Thomas Mikolov (Google, 2013) proposed to use this foundational linguistic principle:
**"Similar words tend to occur together and will have similar contexts."**

- **Context** = the surrounding words of a target word.

Examples:

- *Apples and oranges are fruits.*
- *A mango is a fruit.*

Now the challenge: Find a numerical representation that allows **similar words to have a similar representation.**

---

## Some Basic Concepts

### What is a Vector?

- A vector is a quantity with both magnitude and direction.
  Think of an arrow: length = strength, arrowhead = direction.
- In data science, vectors represent objects as points in *n*-dimensional space.
  Each dimension = one attribute.

**Example:** Suppose we describe an apple in 5 attributes.

| Attribute | Value             | Numerical Mapping |
|-----------|-------------------|-------------------|
| Size      | large             | 0.6 |
| Firmness  | hard              | 0.9 |
| Juiciness | dry               | 0.2 |
| Color     | green             | 0.8 |
| Origin    | Washington State  | 1.0 |

This becomes a **5-dimensional vector** for “apple.”

---

### Visualizing Dimensions

- **1D:** A ruler (length only).
- **2D:** A mousepad (length & width).
- **3D:** A Wii console (length, width, height).
- **4D:** A tesseract (a 4D cube, projected into 3D).
- **>4D:** Impossible for humans to visualize — trivial for computers.

---

### Comparing Vectors

- **Norm (length):** size of the arrow.
- **L1 norm (Manhattan distance):** sum of absolute differences.
- **L2 norm (Euclidean distance):** straight-line distance “as the crow flies.”

Think of:

- Diagonal inside a rectangle (2D).
- Diagonal inside a rectangular prism (3D).

---

### Normalized Vectors

- Divide each component by vector length → **unit vector**.
- Focus on direction (meaning), not scale.
- Magnitude = 1, direction preserved.

Why? This makes vectors easier to compare by direction (semantics) rather than size.

---

### Cosine Similarity

Cosine Similarity is used to calculate the similarity of two vectors.
Formula looks scary, but simplifies dramatically for normalized vectors:

- **Cosine similarity = dot product** (if vectors are normalized).
- **0 → unrelated, 1 → highly similar.**

This is how computers check “closeness” of meaning.

---

## Corpus

Example mini-dataset:

```text
Apple is a fruit
Mango is a fruit
Orange is a fruit
Broccoli is a vegetable
Potato is a vegetable
Spinach is a vegetable
```

Removing filler words ("is", "a") leaves a vocabulary of **8 words.**

---

### One-Hot Encoded Vectors

| Vocabulary | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---------- | - | - | - | - | - | - | - | - |
| apple      | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mango      | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| orange     | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| broccoli   | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| potato     | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| spinach    | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| fruit      | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| vegetable  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

**Problem with one-hot encoding:**

- Vectors grow huge with large vocabularies.
- All words are orthogonal (no sense of similarity).
- Sparse, arbitrary, unhelpful for meaning.

---

### Distributed 2D Vectors (Toy Example)

Imagine we only use **two dimensions** (say, *color* and *juiciness*).
(Using only two dimensions allows for easy visualization.)

- apple     [ 0.08, -0.86]
- mango     [ 0.09, -0.10]
- orange    [ 0.00,  0.90]
- broccoli  [-0.62, -0.86]
- potato    [-0.56,  0.14]
- spinach   [-0.95, -0.70]
- fruit     [ 0.19,  0.69]
- vegetable [ 0.56,  0.29]

This is just a **random** initialization.
Training will **nudge** these distributed vectors so similar words move closer together.

---

### Demo 1: Random Embeddings in 2D

```python
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
```

---

## Neural Network (The Tiny Trainer)

The “Tiny Word Embedding Trainer” is a **very small neural network**.

- **Input:** one word.
- **Output:** predicts the correct related word (fruit or vegetable).
- **Goal:** adjust embeddings so related words move closer together.

Key pieces:

- **Embeddings (`W`)** = the trainable vectors for each word.
- **Forward pass:** look up input word, predict related word.
- **Error:** compare guess vs correct label.
- **Backpropagation:** nudge embeddings to reduce error.

Over time, fruits cluster with fruits, vegetables with vegetables.

---

### Demo 2: Tiny Word Embedding Trainer

```python
"""
Tiny Word Embedding Trainer (fruits & vegetables, inputs-only update)
"""
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
```

**Explanation in plain words:**

- Pick an input word and its correct target.
- Look up the embedding of the input word.
- Multiply by `W.T` to get predictions for all words.
- Compare prediction vs correct one-hot vector.
- Adjust the embedding of the input word slightly toward the correct answer.

Repeat → embeddings converge.

---

### Multiplying Vector × Matrix

Now that we’ve seen random embeddings plotted, let’s look at how multiplying a vector by a matrix actually transforms those coordinates — the very operation hiding inside our tiny trainer.

When you multiply a vector by a matrix, you’re really saying:
“Take my word’s coordinates, and remix them into a new set of coordinates.”

Analogy:

- Word *cat* = vector `[2, 5, 7]`.
- Multiply by a matrix → `[1.2, -0.8]`.
- That’s like translating the word into a new “language of features.”

Takeaway:

- **Vector = list of numbers.**
- **Matrix = grid of numbers.**
- **Multiplying = transformation.**

---

### Demo 3: Adding a 3rd Dimension

We can easily add a third dimension (e.g., *Firmness*), which can still be nicely visualized, making these modifications:

```python
# --- Vocabulary and Corpus ---
vocab = ["apple", "mango", "orange", "fruit", "broccoli", "potato", "spinach", "vegetable", "soft", "firm"]

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
    ("spinach", "soft")
]

N = 3            # embedding dimensions (hidden layer size)
epochs = 50      # training cycles
```

We added two new descriptive words: **“soft”** and **“firm”**, and six new training pairs.

- The embeddings are now **3-dimensional**.
- More training cycles are needed because the network has more to learn.
- Running the training again separates fruit from vegetables **and** groups items by firmness.

This 3D demo prepares us for the leap to **real embeddings like GloVe with 300 dimensions** — far beyond human visualization, but easy for computers.

---

### Model Parameters

When we talk about **model size** and **parameters**:

- **Model size** = how many numbers (weights) the model stores and updates.
- **Parameters** = the actual weights and biases that get learned during training.

For our tiny 3D model:

- **Vocabulary size = 10 words.**
- **Embedding size = 3 dimensions.**
- **Embedding matrix = 10 × 3 = 30 parameters.**

Compare that to something like GloVe (300D, vocab of 400,000+) → **120 million parameters.**
Same idea, just massively scaled.

How many words are in the English language?
Depends on where you look. If you're looking at words in the dictionary, you can turn to Merriam-Webster, which includes around 470,000 English words. Another option is the Oxford English Dictionary (OED), which includes over 600,000 words.

---

### Target Word Training and Softmax

Our toy model keeps things simple.
But in real **Word2Vec** training:

1. **Both input and target words are updated** (not just input).
2. The raw score vector `z` is normalized using **softmax**.

   - Softmax turns scores into probabilities that sum to 1.
   - This ensures the model learns a proper probability distribution over the vocabulary.

So:

- **Toy trainer** = minimal intuition.
- **Real Word2Vec** = efficient training with softmax, negative sampling, and updates to both input and output embeddings.

---

## Wrap-Up (for Part 1)

- Words can be represented as vectors.
- Similar words → similar vectors.
- One-hot encodings don’t capture meaning.
- Distributed vectors (embeddings) **do capture meaning.**
- Neural networks learn these embeddings through training.
- **Demo 1**: Random embeddings (2D).
- **Demo 2**: Tiny trainer nudging embeddings into place.
- **Demo 3**: Adding a third dimension, preparing for high-dimensional embeddings.

Next: scaling this up leads to **Word2Vec, transformers, and eventually ChatGPT.**
