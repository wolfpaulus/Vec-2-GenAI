""" Same word, different meanings (Transformers vs GloVe) """
from sentence_transformers import SentenceTransformer, util

# Load a small pre-trained transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "He sat on the river bank.",
    "She deposited money at the bank.",
    "The shore was calm and peaceful.",
    "The coins were safely stored."
]

# Encode full sentences
vec = model.encode(sentences)

# Cosine similarity
s0 = util.cos_sim(vec[0], vec[2])
s1 = util.cos_sim(vec[0], vec[3])
s2 = util.cos_sim(vec[1], vec[2])
s3 = util.cos_sim(vec[1], vec[3])
print(f"Similarity ((river) bank, shore): {s0.item():.4f}")
print(f"Similarity ((river) bank, coins stored): {s1.item():.4f}")
print(f"Similarity ((money) bank, shore): {s2.item():.4f}")
print(f"Similarity ((money) bank, coins stored): {s3.item():.4f}")
