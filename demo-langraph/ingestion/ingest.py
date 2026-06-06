from __future__ import annotations

import pickle
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = BASE_DIR / "data" / "uploads"
INDEX_PATH = BASE_DIR / "embeddings" / "syllabus.index"
DOCS_PATH = BASE_DIR / "embeddings" / "docs.pkl"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def run_ingestion(data_dir: Path | None = None) -> int:
    """
    Parse all PDFs in data_dir, chunk with overlap, embed, and persist a FAISS index.
    Returns the number of chunks indexed.
    """
    src = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(src.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {src}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks: list[dict] = []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for chunk_text in _splitter.split_text(text):
                    chunks.append({
                        "text": chunk_text,
                        "source": pdf_path.name,
                        "page": page_num,
                    })
        except Exception as exc:
            print(f"Warning: could not process {pdf_path.name}: {exc}")

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    texts = [c["text"] for c in chunks]
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
