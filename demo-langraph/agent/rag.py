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


def retrieve_with_meta(query: str, k: int = 5) -> tuple[str, list[dict]]:
    """
    Return (formatted_context, citations) where citations is a deduplicated list
    of {source, page} dicts in retrieval order — used by the UI to render chips.
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
    citations: list[dict] = []
    seen: set[tuple] = set()

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            parts.append(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}")
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                citations.append({"source": c["source"], "page": c["page"]})

    context = "\n\n---\n\n".join(parts) if parts else "No relevant content found."
    return context, citations


def retrieve(query: str, k: int = 5) -> str:
    """Convenience wrapper — returns formatted context string only."""
    context, _ = retrieve_with_meta(query, k)
    return context
