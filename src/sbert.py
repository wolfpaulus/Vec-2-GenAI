""" Same word, different meanings (Transformers vs GloVe) """
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

# Load a small pre-trained transformer
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)  # use local cache only
sentences = [
    "He sat on the river bank.",
    "She borrowed money from the bank.",
    "The boat was tied to the shore.",
    "The check was safely deposited."
]

# Encode full sentences
vec = model.encode(sentences)

# Reduce to 2D
points = PCA(n_components=2).fit_transform(vec)

# Plot
plt.figure(1)
plt.title("Sentence Embeddings (Transformers)")
for (x, y), label in zip(points, sentences):
    plt.scatter(x, y, c="green", edgecolors="w", linewidth=0.05)
    plt.text(x+0.03, y-0.1, label, fontsize=10)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.tight_layout()

# Cosine similarity
s0 = util.cos_sim(vec[0], vec[2])
s1 = util.cos_sim(vec[0], vec[3])
s2 = util.cos_sim(vec[1], vec[2])
s3 = util.cos_sim(vec[1], vec[3])
print(f"river bank ~ shore: {s0[0][0]:.2f}")
print(f"river bank ~ deposited: {s1[0][0]:.2f}")
print(f"money bank ~ shore: {s2[0][0]:.2f}")
print(f"money bank ~ deposited: {s3[0][0]:.2f}")

# Visualize these similarities in a bar chart
labels = [
    "river bank ~ shore",
    "river bank ~ deposited",
    "money bank ~ shore",
    "money bank ~ deposited"
]
values = [s0[0][0], s1[0][0], s2[0][0], s3[0][0]]
plt.figure(2)
plt.barh(labels, values)
plt.xlabel("Cosine Similarity")
plt.title("Semantic Similarity of 'bank' in Different Contexts")
plt.show()
