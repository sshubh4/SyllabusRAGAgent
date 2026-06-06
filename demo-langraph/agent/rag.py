from __future__ import annotations

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
INDEX_PATH = BASE_DIR / "embeddings" / "syllabus.index"
DOCS_PATH  = BASE_DIR / "embeddings" / "docs.pkl"

# L2 distance threshold for flagging weak retrieval.
# For unit-normalized vectors: L2_dist² = 2 - 2·cos_sim
# dist=1.2 → cos_sim ≈ 0.28  (below this = poor topic match)
_WEAK_MATCH_THRESHOLD = 1.2

from ingestion.ingest import EMBED_MODEL_NAME

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _load_index() -> tuple[faiss.Index, list[dict]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError("No index found. Upload a PDF syllabus to build the index.")
    if not DOCS_PATH.exists():
        raise FileNotFoundError("Chunk store not found. Upload a PDF to rebuild the index.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, "rb") as fh:
        chunks: list[dict] = pickle.load(fh)
    return index, chunks


def retrieve_with_meta(
    query: str, k: int = 10
) -> tuple[str, list[dict], list[dict], bool]:
    """
    Returns:
        context_str      — formatted text with [Source, Page] prefixes for the LLM
        citations        — deduplicated [{source, page}] for UI chips
        retrieval_items  — [{text, source, page, score}] for the context viewer
        low_relevance    — True when best-match L2 distance exceeds threshold
    """
    index, chunks = _load_index()

    if not chunks:
        raise ValueError("Index is empty. Re-upload your syllabus.")

    model    = get_embed_model()
    query_emb = model.encode([query])

    if query_emb.shape[1] != index.d:
        raise ValueError(f"Embedding dimension mismatch: {query_emb.shape[1]} vs {index.d}.")

    k = min(k, len(chunks))
    distances_arr, indices_arr = index.search(query_emb, k)

    best_dist     = float(distances_arr[0][0]) if k > 0 else 99.0
    low_relevance = best_dist > _WEAK_MATCH_THRESHOLD

    parts:           list[str]   = []
    citations:       list[dict]  = []
    retrieval_items: list[dict]  = []
    seen:            set[tuple]  = set()

    for dist, idx in zip(distances_arr[0], indices_arr[0]):
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            # Cosine similarity from L2 distance (unit-normalized embeddings)
            cos_sim = max(0.0, 1.0 - float(dist) ** 2 / 2.0)
            score   = round(cos_sim, 3)

            parts.append(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}")

            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                citations.append({"source": c["source"], "page": c["page"]})

            retrieval_items.append({
                "text":   c["text"],
                "source": c["source"],
                "page":   c["page"],
                "score":  score,
            })

    context = "\n\n---\n\n".join(parts) if parts else "No relevant content found."
    return context, citations, retrieval_items, low_relevance


def retrieve(query: str, k: int = 10) -> str:
    """Convenience wrapper — returns formatted context string only."""
    context, _, _, _ = retrieve_with_meta(query, k)
    return context
