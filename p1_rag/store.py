from functools import lru_cache
from . import config as C

@lru_cache(1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(C.EMBED_MODEL)

def embed(texts):
    return _model().encode(texts, normalize_embeddings=True, batch_size=64).tolist()

def dim():
    return _model().get_sentence_embedding_dimension()

@lru_cache(1)
def collection():
    import chromadb
    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    return client.get_or_create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})
