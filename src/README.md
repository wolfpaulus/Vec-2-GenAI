# Source Code

## Demo 1

Creating Word Embeddings

### [vectors.py](vectors.py)

Showing a Matplotlib scatter-chart of 8 2-dimensional random vector, representing "apple", "mango", "orange", "fruit", "broccoli", "potato", "spinach", and "vegetable".

### [training.py](training.py)

Tiny Word Embedding Trainer (inputs-only update).
Running 6 word-pairs 20 times through the trainer.
Prints the embeddings after training (no plot).

### [training_plot.py](training_plot.py)

Same as training.py but with interactive plotting throughout the training process.

### [word2vec.py](word2vec.py)

Complete word-2-vec algorithm, outputs are updated as well, and softmax is used to turn outputs into predictions.
Same vocabulary and same word-pairs are used.

### [word2vec_3d.py](word2vec_3d.py)

Same as word2vec.py but with a 3rd dimension added.
Vocabulary is now 10 words and 12 word-pairs are used.

## Demo 2

Using a larger embeddings model: GloVe

### [pre-proc.py](pre_proc.py)

This script needs to be called with source and target model file like so:
`python src/pre_proc.py <in_filepath> <out_filepath>`
It pre-processes GloVe embeddings to shorten the vocabulary and normalizing the vectors.
Only those vectors will remain, that can be found in Kaggle's '⅓ Million Most Frequent English Words on the Web'.
All vectors are normalized (divided by their N2 norm)

### [similarity.py](similarity.py)

Find most similar words to a given input word using preprocessed GloVe embeddings.
Usage: python src/similar_words.py <glove_filepath>

### [outlier.py](outlier.py)

Finding the outlier in a list of words using preprocessed GloVe embeddings.

### [analogy.py](analogy.py)

Find the missing word in an analogy using preprocessed GloVe embeddings.
E.g.,

- KING is to QUEEN as MAN is to **WOMAN**
- Berlin is to Germany as Paris is to **France**
- Car is to driver as plane is to **pilot**

## Demo 3

Using Sentence-Transformers

### [download_gpt.py](download_gpt.py)

Downloading a  GPT-2-Large model from Huggingface and saving it to local disk: ../models/gpt2-large_local

### [sbert.py](sbert.py)

Creating sentence embeddings using the "all-MiniLM-L6-v2" model from Sentence-Transformers.

### [q_and_a.py](q_and_a.py)

Simple Q&A system using Sentence-Transformers to find best matching sentence snippet.
Input example questions:

- "Who was the villain?"
- "Who told the story?"

### [heroes.py](heroes.py)

Completing a sentence using GPT-2-Large.
