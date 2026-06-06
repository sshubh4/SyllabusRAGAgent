from __future__ import annotations

import sys
from pathlib import Path

# Ensure the demo-langraph directory is on sys.path so agent/ingestion imports work
# regardless of which working directory Streamlit is launched from.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import streamlit as st

from agent.graph import app as agent_app
from ingestion.ingest import run_ingestion

UPLOAD_DIR = _HERE / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Syllabus RAG Agent", page_icon="📚", layout="wide")
st.title("📚 Syllabus-Aware AI Assistant")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Syllabus")
    uploaded_file = st.file_uploader(
        "Upload a PDF syllabus",
        type=["pdf"],
        help="Chunked and indexed automatically. Answers will cite page numbers.",
    )

    if uploaded_file is not None:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        file_path = UPLOAD_DIR / uploaded_file.name

        if file_key not in st.session_state.processed_files:
            file_path.write_bytes(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

            with st.spinner("Chunking and indexing…"):
                try:
                    n_chunks = run_ingestion(UPLOAD_DIR)
                    st.success(f"Indexed {n_chunks} chunks. Ask away!")
                    st.session_state.processed_files.add(file_key)
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
        else:
            st.info(f"'{uploaded_file.name}' already indexed this session.")

        st.caption(f"{uploaded_file.size / 1024:.1f} KB")

    st.divider()
    st.subheader("Indexed files")
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if pdf_files:
        for p in pdf_files:
            st.text(f"• {p.name}")
    else:
        st.info("No PDFs uploaded yet.")

# ── Chat ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

if query := st.chat_input("Ask about your syllabus, weather (+ city), or anything…"):
    st.chat_message("user").write(query)
    st.session_state.messages.append(("user", query))

    history = [f"{r.capitalize()}: {m}" for r, m in st.session_state.messages[:-1]]

    with st.spinner("Thinking…"):
        try:
            result = agent_app.invoke({"query": query, "messages": history})
            response = result.get("result", "No response generated.")
        except Exception as exc:
            response = f"Error: {exc}"

    st.chat_message("assistant").write(response)
    st.session_state.messages.append(("assistant", response))
