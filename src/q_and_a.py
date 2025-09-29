""" Simple Q&A system using Sentence-BERT """
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences (like snippets from Treasure Island)
sentences = [
    "Jim Hawkins is the narrator of Treasure Island.",
    "Long John Silver is a cunning one-legged pirate.",
    "They set sail to find buried treasure.",
    "The Squire and the Doctor organized the voyage.",
    "The crew had mixed loyalties."
]

# Encode the sentences once
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

print("Ask me questions about Treasure Island! (type 'exit' to quit)\n")
while True:
    # Get user input
    question = input("Question: ")
    if question.lower().strip() in {"exit", "quit"}:
        print("Goodbye!")
        break

    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities, find the print best match
    cos_sim = util.cos_sim(question_embedding, sentence_embeddings)
    best_idx = cos_sim.argmax().item()
    print(f"Best Answer: {sentences[best_idx]}\n")
