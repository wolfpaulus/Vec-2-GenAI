# From Numbers to Meaning: The Building Blocks of Generative AI

**Key Words:** Generative AI, Word Embeddings, Neural Networks, Transformers, Attention Mechanism

---

## Abstract

This talk introduces the core ideas that power today’s generative AI systems, beginning not with algorithms, but with something familiar: numbers. We explore how words can be represented as vectors — lists of numbers that capture the attributes of objects. By comparing these vectors, we begin to see how similarity and relationships emerge: apples and oranges are close, while apples and spinach are not. This grounding in vectors and simple operations provides the intuition for how meaning can be encoded in numbers.

From here, the story builds step by step. We examine how one-hot encodings assign each word its own axis, and why this approach quickly becomes unmanageable. Distributed embeddings are introduced as a more powerful alternative, bringing words with similar contexts closer together. Small demonstrations show how these embeddings evolve, grouping fruits with fruits and vegetables with vegetables. A tiny neural network provides a tangible illustration of how such representations are trained and refined.

Scaling up, we look at pre-trained embeddings such as GloVe, which capture rich relationships in language, allowing for simple word arithmetic like king – man + woman ≈ queen. However, these embeddings are still fixed, unable to adapt when a word’s meaning changes with context. This sets the stage for more advanced models that address ambiguity in language.

The Transformer architecture addresses ambiguity in language. Our focus is not on its inner mechanics but on the broader idea that meaning in language is not fixed but flexible, and can be dynamically adjusted based on context. Examples such as resolving the pronoun “it” differently in nearly identical sentences show how models can shift meaning on the fly:
“The animal didn’t cross the street because it was too tired.” → it = animal
“The animal didn’t cross the street because it was bustling.” → it = street

The conclusion emphasizes that the journey is as important as the destination. By the end of the talk, attendees will not be expected to compute vector–matrix multiplications or implement neural networks. Instead, the key takeaway is that complex language understanding emerges from simple building blocks. Lists of numbers, when compared, nudged, and transformed, can come to represent concepts and relationships. This progression from simple vectors to dynamic context-aware models offers a powerful perspective: generative AI is not magic, but the result of stacking understandable ideas in clever ways.

Attendees will leave with a high-level intuition of how AI builds meaning from numbers. More importantly, they will appreciate how approachable these ideas are, and how they connect directly to the technologies reshaping communication, creativity, and engineering today.

---

## Outline

1. **Introduction**
   - Motivation: Why generative AI matters
   - Objectives of the talk

2. **Vectors and Meaning**
   - Definition of vectors in mathematics and data science
   - Comparing vectors: norms, similarity, cosine similarity
   - Vector × matrix transformations

3. **Word Representations**
   - One-hot encoding and its limitations
   - Distributed vectors and embeddings
   - Training embeddings with simple neural networks

4. **Scaling Up**
   - Introduction to GloVe embeddings
   - Strengths and limitations of pre-trained embeddings
   - Demonstrations: word analogies and similarity

5. **Transformers and Attention**
   - The attention mechanism explained
   - Examples of contextual disambiguation (“it” = animal vs street)
   - Parallel processing and scalability

6. **Wrap-Up**
   - Journey from vectors to Transformers
   - Implications for modern AI systems like ChatGPT
