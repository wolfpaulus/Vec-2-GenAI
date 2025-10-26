""" GloVe Vs GPT2 embeddings for the word 'bank' in different contexts.  """
from transformers import AutoTokenizer, AutoModel
from mdl_utils import load_vectors

# Load GloVe and print first 5 elements of the vector for "bank"
vectors = load_vectors("models/wikigiga_50d.txt")
print(f"Embeddings for 'bank': {vectors['bank'][:5]}\n")

# Load tokenizer and model (offline if already cached)
tokenizer = AutoTokenizer.from_pretrained("./models/gpt2-large_local", local_files_only=True)
model = AutoModel.from_pretrained("./models/gpt2-large_local", local_files_only=True)
print(f"Loaded GPT model and tokenizer. Vocab size: {model.config.vocab_size:,}")
s1 = "I deposited money in the bank"
s2 = "The fisherman sat by the bank"

# Get embeddings for the 'bank' token
e1 = model(**tokenizer(s1, return_tensors="pt")).last_hidden_state[0, s1.split().index("bank")]
e2 = model(**tokenizer(s2, return_tensors="pt")).last_hidden_state[0, s2.split().index("bank")]

print(f"Embedding for 'bank' in sentence 1: {e1[:5]}")
print(f"Embedding for 'bank' in sentence 2: {e2[:5]}")
