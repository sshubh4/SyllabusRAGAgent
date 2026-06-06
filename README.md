# SyllabusRAGAgent

A multi-tool RAG agent that answers questions about course syllabi. Upload a PDF and ask natural-language questions — answers are grounded strictly in the document with **page-level citations**.

Built as a LangGraph state machine that routes each query to one of three handlers via an LLM-based intent classifier: syllabus Q&A, live weather lookup, or general chat.

---

## Demo

Upload `demo-langraph/data/uploads/CIS 155 Syllabus.pdf` (included) and try:

- *"What are the office hours?"*
- *"When is the final exam?"*
- *"What is the late submission policy?"*
- *"What's the weather in Chicago tomorrow?"*

> **Note:** Streamlit Community Cloud containers reset periodically, so the FAISS index is rebuilt on the next PDF upload after a restart.

---

## Architecture

```
User query
    │
    ▼
┌─────────┐   syllabus_rag ──► FAISS retrieve (top-5 chunks) ──► Claude answer + citations
│  Router │──► weather       ──► Open-Meteo geocode + forecast  ──► Claude summary
│ (Claude)│──► chat          ──► Claude conversational reply
└─────────┘
```

Implemented as a LangGraph `StateGraph` with four nodes: `router → (syllabus_rag | weather | chat)`.

---

## How it works

### Ingestion pipeline
1. `pypdf` extracts text page-by-page from every uploaded PDF
2. `RecursiveCharacterTextSplitter` splits on `\n\n → \n → ". " → " "` boundaries  
   (chunk size: 500 chars, overlap: 50 chars — preserves context across split points)
3. Each chunk is stored with metadata: `{text, source filename, page number}`
4. `all-MiniLM-L6-v2` (sentence-transformers, 384-dim, runs locally) embeds all chunks
5. FAISS `IndexFlatL2` persisted to disk as `embeddings/syllabus.index` + `docs.pkl`

### Query pipeline
1. A single 10-token Claude call classifies intent: `syllabus_rag | weather | chat`
2. **Syllabus path**: query is embedded → top-5 nearest chunks retrieved → chunks (with `[Source: file, Page N]` tags) injected into a strict RAG prompt → Claude generates an answer with inline citations
3. **Weather path**: Claude extracts city + date from the query → Open-Meteo free API (no key) → Claude formats the forecast
4. **Chat path**: full conversation history passed to Claude for context-aware replies

---

## Tech stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM | Claude API (Anthropic) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`faiss-cpu`) |
| PDF parsing | pypdf |
| Chunking | langchain-text-splitters |
| Weather | Open-Meteo (free, no key) |

---

## Setup

### Prerequisites

- Python 3.9+
- [Anthropic API key](https://console.anthropic.com/)

### Local

```bash
git clone https://github.com/sshubh4/SyllabusRAGAgent.git
cd SyllabusRAGAgent

cp .env.example demo-langraph/.env
# Edit demo-langraph/.env and set ANTHROPIC_API_KEY=sk-ant-...

cd demo-langraph
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload a PDF from the sidebar — it will be chunked and indexed automatically.

### Streamlit Community Cloud (free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `demo-langraph/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** — Streamlit installs dependencies automatically from `requirements.txt`

---

## Design decisions

**LLM router over keyword matching**  
A 10-token Claude call classifies intent with near-zero latency and cost. It handles synonyms, paraphrasing, and compound questions that a regex approach would miss (e.g. "is there a quiz next Thursday?" routes correctly to `syllabus_rag`).

**LangGraph state machine**  
Clean separation of concerns. Adding a new tool (grade calculator, calendar lookup, etc.) requires one new node and one new conditional edge — no tangled if/else chains.

**Recursive chunking with overlap**  
Splitting on semantic boundaries (`\n\n → \n → ". "`) keeps related sentences together. The 50-char overlap prevents losing context at split points — a sentence that straddles two chunks is still retrievable. Page-per-chunk strategies miss both of these properties.

**Chunk-level metadata and citations**  
Every chunk carries `{source, page}`. The RAG prompt instructs Claude to cite these inline, giving students verifiable references rather than confident-sounding hallucinations.

**FAISS flat index**  
Syllabi are short documents. A flat L2 index does exact nearest-neighbor search with no approximation error. IVF/HNSW indexes are only worth the complexity at hundreds of thousands of vectors.
