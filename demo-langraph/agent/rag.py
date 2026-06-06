from __future__ import annotations

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
INDEX_PATH = BASE_DIR / "embeddings" / "syllabus.index"
DOCS_PATH = BASE_DIR / "embeddings" / "docs.pkl"

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _load_index() -> tuple[faiss.Index, list[dict]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "No index found. Upload a PDF syllabus to build the index."
        )
    if not DOCS_PATH.exists():
        raise FileNotFoundError(
            "Chunk store not found. Upload a PDF syllabus to rebuild the index."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "rb") as fh:
        chunks: list[dict] = pickle.load(fh)
    return index, chunks


def retrieve(query: str, k: int = 5) -> str:
    """
    Return the top-k most relevant chunks as a single string.
    Each chunk block is prefixed with its source filename and page number so
    the LLM can include inline citations in its answer.
    """
    index, chunks = _load_index()

    if not chunks:
        raise ValueError("Index is empty. Re-upload your syllabus.")

    model = get_embed_model()
    query_emb = model.encode([query])

    if query_emb.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_emb.shape[1]}, index={index.d}."
        )

    k = min(k, len(chunks))
    _, indices = index.search(query_emb, k)

    parts: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            parts.append(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}")

    return "\n\n---\n\n".join(parts) if parts else "No relevant content found."
