""" Simple example of using SentenceTransformers and HuggingFace Transformers."""
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
prompt = "Not all heroes"
print("Start:", prompt)

for _ in range(10):  # number of steps to grow the sentence
    outputs = generator(
        prompt,
        max_new_tokens=1,   # just one token at a time
        num_return_sequences=1,
        do_sample=False     # always take the most likely next token
    )
    prompt = outputs[0]["generated_text"]
    print("→", prompt)
