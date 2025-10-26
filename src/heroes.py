""" Simple example of using SentenceTransformers and HuggingFace Transformers."""
from transformers import pipeline

generator = pipeline("text-generation", model="./models/gpt2-large_local")  # 3.25GB
prompt = "Not all heroes wear"
print(f"Start: {prompt}")

for _ in range(15):  # number of steps to grow the sentence
    outputs = generator(prompt,
                        max_new_tokens=1,        # just one token at a time
                        pad_token_id=50256,      # for open-end generation.
                        num_return_sequences=1,  # just one sequence
                        temperature=0.1)         # some randomness)
    prompt = outputs[0]["generated_text"]
    print("→", prompt)
