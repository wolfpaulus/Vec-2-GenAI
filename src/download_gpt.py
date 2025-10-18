""" Download GPT model for local use. """
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model you want to download
model_name = "gpt2-large"

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained("./models/gpt2-large_local")

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("./models/gpt2-large_local")
