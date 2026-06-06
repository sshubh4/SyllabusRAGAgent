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

UPLOAD_DIR = _HERE / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Syllabus AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & globals ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main {
    background-color: #0f1117 !important;
    color: #e4e6eb;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stDecoration"],
[data-testid="stHeader"],
#MainMenu,
footer { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #161922 !important;
    border-right: 1px solid #2a2e3a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #e4e6eb !important; }

.sidebar-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b8f9a !important;
    margin: 16px 0 8px 0;
}

[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid #2a2e3a !important;
    margin: 12px 0;
}

.sidebar-footer {
    font-size: 0.75rem;
    color: #8b8f9a;
    padding-top: 12px;
    border-top: 1px solid #2a2e3a;
    margin-top: 8px;
    line-height: 1.6;
}
.sidebar-footer a { color: #6366f1; text-decoration: none; }
.sidebar-footer a:hover { text-decoration: underline; }

/* Indexed file chips */
.file-chip {
    display: block;
    background: #1c1f2b;
    border: 1px solid #2a2e3a;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #8b8f9a !important;
    margin: 3px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1c1f2b !important;
    border: 1px dashed #2a2e3a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #8b8f9a !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1c1f2b !important;
    border: 1px solid #2a2e3a !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #8b8f9a !important;
    font-size: 0.78rem !important;
}

/* Architecture diagram */
.diagram-block {
    background: #0f1117;
    border: 1px solid #2a2e3a;
    border-radius: 6px;
    padding: 12px 14px;
    font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace;
    font-size: 0.72rem;
    color: #8b8f9a;
    line-height: 1.75;
    white-space: pre;
    overflow-x: auto;
    margin: 0;
}

/* ── Main header ── */
.main-header { padding: 24px 0 14px 0; }
.main-header h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: #e4e6eb;
    margin: 0 0 6px 0;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.main-header .subtitle {
    font-size: 0.85rem;
    color: #8b8f9a;
    line-height: 1.65;
    max-width: 700px;
    margin: 0;
}
.main-divider {
    border: none;
    border-top: 1px solid #2a2e3a;
    margin: 16px 0 20px 0;
}

/* ── Message layout ── */
[data-testid="stMarkdownContainer"] { width: 100%; }

.msg-wrapper {
    display: flex;
    margin-bottom: 14px;
    width: 100%;
}
.msg-wrapper.user      { justify-content: flex-end; }
.msg-wrapper.assistant { justify-content: flex-start; }

/* ── Bubbles ── */
.msg-bubble {
    max-width: 78%;
    padding: 12px 16px;
    font-size: 0.875rem;
    line-height: 1.65;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.msg-bubble.user {
    background: #6366f1;
    color: #ffffff;
    border-radius: 14px 14px 4px 14px;
}
.msg-bubble.assistant {
    background: #1c1f2b;
    border: 1px solid #2a2e3a;
    border-left: 3px solid #10b981;
    color: #e4e6eb;
    border-radius: 4px 14px 14px 14px;
}
.msg-bubble code {
    background: #0f1117;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.82em;
    font-family: 'SFMono-Regular', 'Consolas', monospace;
    color: #a5b4fc;
}
.msg-bubble ul {
    margin: 4px 0 4px 18px;
    padding: 0;
}
.msg-bubble li { margin-bottom: 2px; }

/* ── Meta row (badge + citation chips) ── */
.msg-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #2a2e3a;
}

/* ── Route badges ── */
.route-badge {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 3px;
    line-height: 1.6;
}
.badge-syllabus_rag { background: #312e81; color: #a5b4fc; }
.badge-weather      { background: #1e3a5f; color: #93c5fd; }
.badge-chat         { background: #1f2937; color: #9ca3af; }

/* ── Citation chips ── */
.citation-chip {
    display: inline-block;
    font-size: 0.68rem;
    color: #8b8f9a;
    border: 1px solid #2a2e3a;
    border-radius: 3px;
    padding: 1px 7px;
    background: #0f1117;
    line-height: 1.6;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #1c1f2b !important;
    border: 1px solid #2a2e3a !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.18) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e4e6eb !important;
    font-size: 0.875rem !important;
    caret-color: #6366f1 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8b8f9a !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 0.82rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #6366f1 !important;
    font-size: 0.82rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a2e3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4254; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Minimal markdown to HTML for chat bubble content."""
    t = _html.escape(text)
    # Bold
    t = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", t, flags=re.DOTALL)
    # Italic (not preceded/followed by *)
    t = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    # Inline code
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    # Unordered list blocks
    def _ul_block(m: re.Match) -> str:
        items = re.findall(r"^[-*]\s+(.+)$", m.group(), re.MULTILINE)
        lis = "".join(f"<li>{i}</li>" for i in items)
        return f"<ul>{lis}</ul>"
    t = re.sub(r"(?:^[-*]\s+.+\n?)+", _ul_block, t, flags=re.MULTILINE)
    # Paragraph breaks then single line breaks
    t = re.sub(r"\n{2,}", "<br><br>", t)
    t = t.replace("\n", "<br>")
    return t


_BADGE_META: dict[str, tuple[str, str]] = {
    "syllabus_rag": ("SYLLABUS RAG", "badge-syllabus_rag"),
    "weather":      ("WEATHER",      "badge-weather"),
    "chat":         ("CHAT",         "badge-chat"),
}


def _render_user_message(content: str) -> None:
    st.markdown(
        f'<div class="msg-wrapper user">'
        f'<div class="msg-bubble user">{_html.escape(content)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_assistant_message(
    content: str, route: str, citations: list[dict]
) -> None:
    label, badge_cls = _BADGE_META.get(route, ("CHAT", "badge-chat"))
    badge_html = f'<span class="route-badge {badge_cls}">{label}</span>'

    chips_html = "".join(
        f'<span class="citation-chip">'
        f'{_html.escape(c.get("source", "").removesuffix(".pdf"))}'
        f' &middot; p.{c.get("page", "?")}'
        f"</span>"
        for c in citations
    )

    meta_html = f'<div class="msg-meta">{badge_html}{chips_html}</div>'

    st.markdown(
        f'<div class="msg-wrapper assistant">'
        f'<div class="msg-bubble assistant">'
        f"<div>{_md_to_html(content)}</div>"
        f"{meta_html}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-section-label">Upload Syllabus</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Chunked and indexed automatically. Answers cite page numbers.",
    )

    if uploaded_file is not None:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        file_path = UPLOAD_DIR / uploaded_file.name

        if file_key not in st.session_state.processed_files:
            file_path.write_bytes(uploaded_file.getbuffer())
            with st.spinner("Indexing..."):
                try:
                    n_chunks = run_ingestion(UPLOAD_DIR)
                    st.success(f"Indexed {n_chunks} chunks")
                    st.session_state.processed_files.add(file_key)
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
        else:
            st.info("Already indexed this session")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-label">Indexed Files</p>', unsafe_allow_html=True)

    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if pdf_files:
        st.markdown(
            "".join(
                f'<span class="file-chip">{_html.escape(p.name)}</span>'
                for p in pdf_files
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-size:0.78rem;color:#8b8f9a">No files uploaded yet</span>',
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

    st.markdown(
        "<div class='sidebar-footer'>"
        "Built by Shubham Sharma &nbsp;&middot;&nbsp;"
        '<a href="https://github.com/sshubh4/SyllabusRAGAgent" target="_blank">GitHub</a>'
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>"
    "<h1>Syllabus AI Assistant</h1>"
    "<p class='subtitle'>Ask questions about your syllabus — answers grounded in the "
    "document with page citations. Powered by a LangGraph agent that routes between "
    "document QA, weather, and chat.</p>"
    "</div>"
    "<hr class='main-divider'>",
    unsafe_allow_html=True,
)


# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        _render_user_message(msg["content"])
    else:
        _render_assistant_message(
            msg["content"],
            msg.get("route", "chat"),
            msg.get("citations") or [],
        )


# ── Input ──────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about your syllabus, weather, or anything..."):
    _render_user_message(query)
    st.session_state.messages.append({"role": "user", "content": query})

    history = [
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    ]

    with st.spinner("Thinking..."):
        try:
            result    = agent_app.invoke({"query": query, "messages": history})
            response  = result.get("result", "No response generated.")
            route     = result.get("tool", "chat")
            citations = result.get("citations") or []
        except Exception as exc:
            response, route, citations = f"Error: {exc}", "chat", []

    _render_assistant_message(response, route, citations)
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response,
        "route":     route,
        "citations": citations,
    })
