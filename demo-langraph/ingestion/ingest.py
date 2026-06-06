from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# QA-optimised model: handles asymmetric question→document retrieval
# far better than general-purpose all-MiniLM-L6-v2
EMBED_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

BASE_DIR = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = BASE_DIR / "data" / "uploads"
INDEX_PATH = BASE_DIR / "embeddings" / "syllabus.index"
DOCS_PATH  = BASE_DIR / "embeddings" / "docs.pkl"

# Larger chunks + more overlap to avoid cutting policies mid-sentence
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)


def _clean_text(text: str) -> str:
    """
    Remove common PyMuPDF extraction artifacts:
    - Collapse 3+ blank lines → 2
    - Collapse repeated spaces/tabs → single space
    - Drop lines that are only symbols/punctuation (no meaningful word content)
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Keep only lines with at least 3 alphanumeric characters
        if len(re.sub(r"[^\w]", "", line)) >= 3:
            lines.append(line)
    return "\n".join(lines)


def _extract_page_text(page) -> str:
    """Extract and clean text from a single PyMuPDF page."""
    raw = page.get_text("text")
    return _clean_text(raw)


def _page_header(content: str) -> str:
    """
    Return the first short, meaningful line of a page to use as a section label.
    Helps the embedding model associate chunks with their document section.
    """
    for line in content.split("\n"):
        line = line.strip()
        if 5 <= len(line) <= 90:
            return line
    return ""


def run_ingestion(data_dir: Path | None = None) -> int:
    """
    Parse all PDFs with PyMuPDF (best text fidelity for complex/multi-column layouts),
    clean artifacts, chunk with overlap, embed, and persist a FAISS index.
    Returns the number of chunks indexed.
    """
    src = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(src.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {src}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    chunks: list[dict] = []

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(doc.page_count):
                content = _extract_page_text(doc.load_page(page_num))
                if not content.strip():
                    continue
                header = _page_header(content)
                for i, chunk_text in enumerate(_splitter.split_text(content)):
                    # Prefix later chunks with the section header so the embedding
                    # model can associate them with the right page context even when
                    # the header text isn't repeated in the chunk body.
                    if header and i > 0 and not chunk_text.startswith(header):
                        embed_text = f"{header}: {chunk_text}"
                    else:
                        embed_text = chunk_text
                    chunks.append({
                        "text":       chunk_text,   # original — shown to Claude & in context viewer
                        "embed_text": embed_text,   # prefixed — used only for embedding
                        "source":     pdf_path.name,
                        "page":       page_num + 1,
                    })
            doc.close()
        except Exception as exc:
            print(f"Warning: could not process {pdf_path.name}: {exc}")

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    texts      = [c["embed_text"] for c in chunks]
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
