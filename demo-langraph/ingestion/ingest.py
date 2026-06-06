from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = BASE_DIR / "data" / "uploads"
INDEX_PATH = BASE_DIR / "embeddings" / "syllabus.index"
DOCS_PATH  = BASE_DIR / "embeddings" / "docs.pkl"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def _extract_page(page) -> str:
    """
    Extract all readable content from a single pdfplumber Page:
    - Tables are formatted as pipe-delimited rows so structure is preserved
    - Body text is extracted with layout awareness (handles multi-column)
    Images and embedded graphics are not extractable as text by any
    pure-Python PDF library; that content is silently skipped.
    """
    parts: list[str] = []

    # ── Tables ────────────────────────────────────────────────────────────
    # Extract before body text so we can render them as structured rows
    # rather than the garbled run-together text pypdf produces.
    for table in page.extract_tables():
        if not table:
            continue
        rows = []
        for row in table:
            if not row:
                continue
            cells = [str(c or "").strip() for c in row]
            if any(cells):                          # skip fully empty rows
                rows.append(" | ".join(cells))
        if rows:
            parts.append("\n".join(rows))

    # ── Body text ─────────────────────────────────────────────────────────
    # layout=True preserves column order and spacing better than the default
    text = page.extract_text(layout=True) or ""
    if text.strip():
        parts.append(text)

    return "\n\n".join(parts)


def run_ingestion(data_dir: Path | None = None) -> int:
    """
    Parse all PDFs in data_dir, extract text + tables with pdfplumber,
    chunk with overlap, embed with sentence-transformers, persist FAISS index.
    Returns the number of chunks indexed.
    """
    src = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(src.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {src}")

    model  = SentenceTransformer("all-MiniLM-L6-v2")
    chunks: list[dict] = []

    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    content = _extract_page(page)
                    if not content.strip():
                        continue
                    for chunk_text in _splitter.split_text(content):
                        chunks.append({
                            "text":   chunk_text,
                            "source": pdf_path.name,
                            "page":   page_num,
                        })
        except Exception as exc:
            print(f"Warning: could not process {pdf_path.name}: {exc}")

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index size mismatch: {index.ntotal} vectors for {len(chunks)} chunks."
        )

    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)

    return len(chunks)


if __name__ == "__main__":
    n = run_ingestion()
    print(f"Indexed {n} chunks → {INDEX_PATH}")
