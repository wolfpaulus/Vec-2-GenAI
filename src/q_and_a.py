""" Simple Q&A system using Sentence-Transformers to find best matching snippet. """
from sentence_transformers import SentenceTransformer, util

# Load model, ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

# Example sentences (like snippets from Treasure Island)
sentences = [
    "Jim Hawkins is the narrator of Treasure Island.",
    "They set sail to find buried treasure.",
    "The Squire and the Doctor organized the voyage.",
    "Long John Silver is a cunning one-legged pirate.",
    "The crew had mixed loyalties."
]

# Encode the sentences once
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

print("Ask me questions about Treasure Island! (type 'exit' to quit)\n")
while True:
    # Get user input
    question = input("Question: ")
    if question.lower().strip() in {"exit", "quit"}:
        exit()

    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities, find the print best match
    cos_sim = util.cos_sim(question_embedding, sentence_embeddings)
    best_idx = int(cos_sim.argmax().item())
    print(f"Best Answer: {sentences[best_idx]}\n")
