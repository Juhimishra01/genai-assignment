# Retrieval-Augmented Generation (RAG) Basics

Retrieval-Augmented Generation, or RAG, is a technique that gives a language model access to external documents at answer time. Instead of relying only on what the model memorised during training, a RAG system first searches a document collection and then passes the most relevant passages to the model along with the question.

## The three stages

A typical RAG pipeline has three stages. In the ingestion stage, documents are split into chunks, converted into embeddings, and stored in a vector database. In the retrieval stage, the user's question is embedded and the top-k most similar chunks are fetched. In the generation stage, the language model reads those chunks and writes an answer that cites them.

## Why chunking matters

Chunk size is one of the most important settings in a RAG system. Chunks that are too small lose context, while chunks that are too large dilute the embedding with unrelated text. A common starting point is 800 characters with an overlap of 120 characters, so that a sentence cut at a boundary still appears whole in the neighbouring chunk.

## Reducing hallucination

A well designed RAG system must say "I don't know" when the retrieved context does not contain the answer. One simple method is a similarity threshold: if no chunk scores above the threshold, the system skips the language model and returns a refusal. Another method is to instruct the model to answer only from the supplied context and to cite chunk numbers for every claim.
