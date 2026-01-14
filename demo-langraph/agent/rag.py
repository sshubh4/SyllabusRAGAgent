import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "embeddings/syllabus.index"
DOCS_PATH = "embeddings/docs.pkl"

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}. "
# run 'python ingestion/ingest.py' to create the index."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Documents file not found: {DOCS_PATH}. "
        )
    
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load index or documents: {str(e)}. "
        ) from e
    
    
    return index, docs

def retrieve(query: str, k: int = 5) -> str:
    try:
        index, docs = load_index()
        
        if len(docs) == 0:
            raise ValueError(
                "No syllabus content available. The index is empty. "
            )
        
        model = get_embed_model()
        query_emb = model.encode([query])
        
        #  embedding dimensions match
        if query_emb.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch"
            )
        
        k = min(k, len(docs))
        _, indices = index.search(query_emb, k)
        
        # Extract retrieved documents
        retrieved_docs = []
        for idx in indices[0]:
            if 0 <= idx < len(docs):
                retrieved_docs.append(docs[idx])
        
        if not retrieved_docs:
            return "No relevant content found"
        
        return "\n\n".join(retrieved_docs)
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise
    except Exception as e:
        # Catch-all
        raise RuntimeError(
            f"Unexpected error during retrieval. retry your upload process "
        ) from e

