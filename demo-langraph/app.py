from __future__ import annotations

import html as _html
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import streamlit as st

from agent.graph import app as agent_app
from ingestion.ingest import run_ingestion

UPLOAD_DIR  = _HERE / "data" / "uploads"
_INDEX_PATH = _HERE / "embeddings" / "syllabus.index"
_DOCS_PATH  = _HERE / "embeddings" / "docs.pkl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Syllabus AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main {
    background-color: #080b10 !important;
    color: #c4cfe4;
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── Header strip — keep visible for sidebar toggle ── */
[data-testid="stHeader"] {
    background: #080b10 !important;
    border-bottom: 1px solid #1c2535 !important;
    height: 3rem !important;
}
[data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer        { display: none !important; }

/* Sidebar toggle buttons */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"]  button {
    color: #556070 !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"]  button:hover {
    color: #00e5b0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #0c0f16 !important;
    border-right: 1px solid #1c2535 !important;
}
[data-testid="stSidebar"] *:not(button):not(svg) { color: #c4cfe4 !important; }

/* Sidebar section label */
.sb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #556070 !important;
    margin: 18px 0 8px 0;
    display: block;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #1c2535 !important;
    margin: 10px 0 !important;
}

/* File row */
.file-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0f1520;
    border: 1px solid #1c2535;
    border-left: 2px solid #1e3a5f;
    border-radius: 3px;
    padding: 5px 8px;
    margin: 3px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8ba0c0 !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Delete button inside sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #1c2535 !important;
    color: #556070 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    padding: 2px 6px !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.4 !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #c0392b !important;
    color: #e74c3c !important;
    background: rgba(192,57,43,0.08) !important;
}

/* Persistence note */
.persist-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3a4d60 !important;
    margin-top: 6px;
    line-height: 1.5;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1520 !important;
    border: 1px dashed #1c2535 !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #556070 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0f1520 !important;
    border: 1px solid #1c2535 !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary {
    color: #556070 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* Architecture diagram */
.diagram-block {
    background: #080b10;
    border: 1px solid #1c2535;
    border-radius: 3px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #3d6b8a;
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
    margin: 0;
}
.diagram-block .hl { color: #00e5b0; }

/* Sidebar footer */
.sb-footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a4d60 !important;
    padding-top: 10px;
    border-top: 1px solid #1c2535;
    margin-top: 10px;
    line-height: 1.8;
}
.sb-footer a { color: #2a7ab8; text-decoration: none; }
.sb-footer a:hover { color: #00e5b0; }

/* ── Main header ── */
.main-header {
    padding: 20px 0 12px 0;
    border-bottom: 1px solid #1c2535;
    margin-bottom: 24px;
}
.main-header .title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #e4eaf6;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.main-header .title .accent { color: #00e5b0; }
.main-header .subtitle {
    font-size: 0.82rem;
    color: #556070;
    line-height: 1.6;
    max-width: 680px;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Chat message layout ── */
[data-testid="stMarkdownContainer"] { width: 100%; }

.msg-wrap {
    display: flex;
    width: 100%;
    margin-bottom: 16px;
    align-items: flex-start;
    gap: 0;
}
.msg-wrap.user      { justify-content: flex-end; }
.msg-wrap.assistant { justify-content: flex-start; }

/* ── Bubbles ── */
.msg-bubble {
    max-width: 76%;
    padding: 11px 15px;
    font-size: 0.875rem;
    line-height: 1.7;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* User: right-aligned, blue, sharp top-right corner */
.msg-bubble.user {
    background: #1a3a7a;
    border: 1px solid #2a5aaa;
    color: #d0e4ff;
    border-radius: 8px 2px 8px 8px;
    font-family: 'Inter', sans-serif;
}

/* Assistant: left-aligned, dark card, teal left accent */
.msg-bubble.assistant {
    background: #0f1928;
    border: 1px solid #1c2d45;
    border-left: 2px solid #00e5b0;
    color: #c4cfe4;
    border-radius: 2px 8px 8px 8px;
    font-family: 'Inter', sans-serif;
}

/* Code inside bubbles */
.msg-bubble code {
    background: #080b10;
    padding: 1px 5px;
    border-radius: 2px;
    font-size: 0.8em;
    font-family: 'JetBrains Mono', monospace;
    color: #00e5b0;
    border: 1px solid #1c2535;
}
.msg-bubble ul {
    margin: 5px 0 5px 18px;
    padding: 0;
}
.msg-bubble li { margin-bottom: 3px; }
.msg-bubble strong { color: #e4eaf6; }

/* ── Meta row ── */
.msg-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    margin-top: 9px;
    padding-top: 8px;
    border-top: 1px solid #1c2d45;
}

/* ── Route badge ── */
.route-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 2px;
    line-height: 1.7;
}
.badge-syllabus_rag { background: #2d1f5e; color: #9f7aea; border: 1px solid #4a2f9a; }
.badge-weather      { background: #0f2a4a; color: #63b3ed; border: 1px solid #1a4a7a; }
.badge-chat         { background: #0f1928; color: #4a6a8a; border: 1px solid #1c2d45; }

/* ── Citation chips ── */
.citation-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #3a6080;
    border: 1px solid #1c2535;
    border-radius: 2px;
    padding: 1px 6px;
    background: #080b10;
    line-height: 1.7;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0f1520 !important;
    border: 1px solid #1c2535 !important;
    border-radius: 5px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #00e5b0 !important;
    box-shadow: 0 0 0 2px rgba(0,229,176,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #c4cfe4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    caret-color: #00e5b0 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #2a3d50 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #00e5b0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Welcome screen ── */
.welcome {
    margin: 40px auto 32px auto;
    max-width: 600px;
    text-align: center;
}
.welcome-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00e5b0;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}
.welcome-body {
    font-size: 0.85rem;
    color: #556070;
    line-height: 1.7;
    margin-bottom: 28px;
    font-family: 'JetBrains Mono', monospace;
}
.welcome-caps {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 32px;
}
.cap {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 10px;
    border: 1px solid #1c2535;
    border-radius: 2px;
    color: #3a6080;
    background: #0f1520;
}
.sample-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a4d60;
    margin-bottom: 10px;
}

/* Sample question buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #0f1520 !important;
    border: 1px solid #1c2535 !important;
    color: #8ba0c0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 8px 12px !important;
    border-radius: 3px !important;
    width: 100% !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.5 !important;
    transition: border-color 0.15s, color 0.15s !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: #00e5b0 !important;
    color: #00e5b0 !important;
    background: #0a1a18 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #1c2535; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2a3d50; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for chat bubble content."""
    t = _html.escape(text)
    t = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", t, flags=re.DOTALL)
    t = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    def _ul(m: re.Match) -> str:
        items = re.findall(r"^[-*]\s+(.+)$", m.group(), re.MULTILINE)
        return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    t = re.sub(r"(?:^[-*]\s+.+\n?)+", _ul, t, flags=re.MULTILINE)
    t = re.sub(r"\n{2,}", "<br><br>", t)
    t = t.replace("\n", "<br>")
    return t


_BADGE_META: dict[str, tuple[str, str]] = {
    "syllabus_rag": ("SYLLABUS RAG", "badge-syllabus_rag"),
    "weather":      ("WEATHER",      "badge-weather"),
    "chat":         ("CHAT",         "badge-chat"),
}


def _render_user(content: str) -> None:
    st.markdown(
        f'<div class="msg-wrap user">'
        f'<div class="msg-bubble user">{_html.escape(content)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_assistant(content: str, route: str, citations: list[dict]) -> None:
    label, badge_cls = _BADGE_META.get(route, ("CHAT", "badge-chat"))
    badge  = f'<span class="route-badge {badge_cls}">{label}</span>'
    chips  = "".join(
        f'<span class="citation-chip">'
        f'{_html.escape(c.get("source","").removesuffix(".pdf"))}'
        f' / p.{c.get("page","?")}'
        f"</span>"
        for c in citations
    )
    meta = f'<div class="msg-meta">{badge}{chips}</div>'
    st.markdown(
        f'<div class="msg-wrap assistant">'
        f'<div class="msg-bubble assistant">'
        f"<div>{_md_to_html(content)}</div>"
        f"{meta}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _delete_pdf(pdf_path: Path) -> None:
    """Delete a PDF and rebuild the index (or clear it if no PDFs remain)."""
    pdf_path.unlink(missing_ok=True)
    # Drop from processed cache so it can be re-uploaded later
    key_prefix = f"{pdf_path.name}_"
    st.session_state.processed_files = {
        k for k in st.session_state.get("processed_files", set())
        if not k.startswith(key_prefix)
    }
    remaining = sorted(UPLOAD_DIR.glob("*.pdf"))
    if remaining:
        try:
            run_ingestion(UPLOAD_DIR)
        except Exception:
            pass
    else:
        for f in (_INDEX_PATH, _DOCS_PATH):
            f.unlink(missing_ok=True)
    st.rerun()


# ── Auto-index on startup ──────────────────────────────────────────────────
# If PDFs exist from a previous session but the index was lost (e.g. after
# a Streamlit Cloud container reset), silently rebuild it.
_pdf_files_now = sorted(UPLOAD_DIR.glob("*.pdf"))
if _pdf_files_now and not _INDEX_PATH.exists():
    with st.spinner("Rebuilding index..."):
        try:
            run_ingestion(UPLOAD_DIR)
        except Exception:
            pass


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sb-label">Upload Syllabus</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Chunked and indexed automatically. Answers cite page numbers.",
    )

    if uploaded_file is not None:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        file_key  = f"{uploaded_file.name}_{uploaded_file.size}"
        file_path = UPLOAD_DIR / uploaded_file.name

        if file_key not in st.session_state.processed_files:
            file_path.write_bytes(uploaded_file.getbuffer())
            with st.spinner("Indexing..."):
                try:
                    n = run_ingestion(UPLOAD_DIR)
                    st.success(f"Indexed {n} chunks")
                    st.session_state.processed_files.add(file_key)
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        else:
            st.info("Already indexed")

    # ── Indexed files with delete ──────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Indexed Files</span>', unsafe_allow_html=True)

    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if pdf_files:
        for p in pdf_files:
            col_name, col_del = st.columns([8, 1])
            with col_name:
                st.markdown(
                    f'<div class="file-row">{_html.escape(p.name)}</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("x", key=f"del_{p.stem}", help=f"Remove {p.name}"):
                    _delete_pdf(p)

        st.markdown(
            '<p class="persist-note">'
            "Files are shared across sessions.<br>"
            "Resets on redeployment."
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-family:\'JetBrains Mono\',monospace;'
            'font-size:0.72rem;color:#3a4d60">No files uploaded yet</span>',
            unsafe_allow_html=True,
        )

    # ── How it works ──────────────────────────────────────────────────────
    # ── Extraction capabilities ────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<span class="sb-label">What gets extracted</span>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.68rem;"
        "line-height:1.9;color:#556070'>"
        "<span style='color:#00e5b0'>&#10003;</span> Body text<br>"
        "<span style='color:#00e5b0'>&#10003;</span> Tables &amp; grids<br>"
        "<span style='color:#00e5b0'>&#10003;</span> Bullet lists &amp; schedules<br>"
        "<span style='color:#c0392b'>&#10007;</span> Images &amp; charts (not extractable)<br>"
        "<span style='color:#c0392b'>&#10007;</span> Scanned / image-only PDFs"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("How it works"):
        st.markdown(
            "<div class='diagram-block'>"
            "Your question\n"
            "      |\n"
            "      v\n"
            "LLM Router\n"
            "(Claude classifies intent)\n"
            "      |\n"
            "  +---+-------+-------+\n"
            "  |           |       |\n"
            "  v           v       v\n"
            "Syllabus    Weather  Chat\n"
            "RAG           API\n"
            "  |\n"
            "  v\n"
            "FAISS top-5 chunks\n"
            "  |\n"
            "  v\n"
            "Claude answer\n"
            "+ page citations"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='sb-footer'>"
        "Built by Shubham Sharma<br>"
        '<a href="https://github.com/sshubh4/SyllabusRAGAgent" target="_blank">'
        "github.com/sshubh4/SyllabusRAGAgent"
        "</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>"
    "<div class='title'><span class='accent'>//</span> Syllabus AI</div>"
    "<div class='subtitle'>"
    "RAG agent &mdash; document QA with page citations &bull; weather &bull; chat"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)


# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# Welcome screen — shown only when chat is empty
if not st.session_state.messages:
    st.markdown(
        "<div class='welcome'>"
        "<div class='welcome-title'>// Ready</div>"
        "<div class='welcome-body'>"
        "Upload a syllabus PDF from the sidebar, then ask anything.<br>"
        "Answers are grounded in the document with page citations."
        "</div>"
        "<div class='welcome-caps'>"
        "<div class='cap'>Text extraction</div>"
        "<div class='cap'>Table parsing</div>"
        "<div class='cap'>Page citations</div>"
        "<div class='cap'>LLM routing</div>"
        "</div>"
        "<div class='sample-label'>Try asking</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _SAMPLES = [
        "What are all the assignment due dates?",
        "What is the grading breakdown?",
        "What is the late submission policy?",
    ]
    cols = st.columns(3)
    for col, q in zip(cols, _SAMPLES):
        with col:
            if st.button(q, key=f"sample_{q[:12]}"):
                st.session_state["pending_query"] = q
                st.rerun()
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user(msg["content"])
        else:
            _render_assistant(
                msg["content"],
                msg.get("route", "chat"),
                msg.get("citations") or [],
            )


# ── Input ──────────────────────────────────────────────────────────────────
typed   = st.chat_input("Ask about your syllabus, weather, or anything...")
pending = st.session_state.get("pending_query")
if pending:
    del st.session_state["pending_query"]
query = typed or pending

if query:
    _render_user(query)
    st.session_state.messages.append({"role": "user", "content": query})

    history = [
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    ]

    with st.spinner("thinking..."):
        try:
            result    = agent_app.invoke({"query": query, "messages": history})
            response  = result.get("result", "No response generated.")
            route     = result.get("tool", "chat")
            citations = result.get("citations") or []
        except Exception as exc:
            response, route, citations = f"Error: {exc}", "chat", []

    _render_assistant(response, route, citations)
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response,
        "route":     route,
        "citations": citations,
    })
