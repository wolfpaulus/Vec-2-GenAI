#!/bin/bash

# Download and preprocess GloVe embeddings Wikipedia + Gigaword 2024, 50d
wget --directory-prefix=models https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip
unzip ./models/glove.2024.wikigiga.50d.zip -d ./models
python3 ./src/pre_proc.py ./models/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt ./models/wikigiga_50d.txt

# Download and preprocess GloVe embeddings Wikipedia + Gigaword 2024, 300d
wget --directory-prefix=models https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip
unzip ./models/glove.2024.wikigiga.300d.zip -d ./models
python3 ./src/pre_proc.py ./models/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt ./models/wikigiga_300d.txt

# Download and preprocess GloVe embeddings Dolma 2024, 300d
wget --directory-prefix=models https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip
unzip ./models/glove.2024.dolma.300d.zip -d ./models
python3 ./src/pre_proc.py ./models/dolma_300_2024_1.2M.100_combined.txt ./models/dolma_300d.txt
