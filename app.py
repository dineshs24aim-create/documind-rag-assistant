"""Streamlit frontend for the Document Q&A Assistant."""

from __future__ import annotations

import hashlib
import html
import os

import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_core import answer_question, process_document


st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --ink: #07060b;
        --panel: #100d18;
        --panel-2: #171222;
        --line: #2c2340;
        --violet: #8b5cf6;
        --violet-bright: #a78bfa;
        --magenta: #d946ef;
        --text: #f7f3ff;
        --muted: #aaa1ba;
    }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 78% 2%, rgba(124, 58, 237, .20), transparent 30rem),
            radial-gradient(circle at 18% 76%, rgba(217, 70, 239, .08), transparent 25rem),
            var(--ink);
    }
    [data-testid="stHeader"] {background: rgba(7, 6, 11, .82); backdrop-filter: blur(16px);}
    [data-testid="stToolbar"] {color: var(--text);}
    [data-testid="stSidebar"] {
        background: #0b0910;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {color: var(--text);}
    [data-testid="stSidebarContent"] {padding-top: 1.35rem;}
    .block-container {max-width: 1280px; padding-top: 2.25rem; padding-bottom: 7rem;}
    h1, h2, h3, p, li, label, [data-testid="stMarkdownContainer"] {color: var(--text);}
    .sidebar-brand {
        display: flex; align-items: center; gap: .75rem; padding: .2rem 0 1.1rem;
    }
    .brand-mark {
        display: grid; place-items: center; width: 42px; height: 42px;
        border-radius: 13px; font-size: 1.25rem; font-weight: 900;
        background: linear-gradient(145deg, var(--violet), var(--magenta));
        box-shadow: 0 0 30px rgba(139, 92, 246, .32);
    }
    .brand-name {font-weight: 800; letter-spacing: -.02em;}
    .brand-sub {font-size: .76rem; color: var(--muted); margin-top: .08rem;}
    .hero {
        position: relative; overflow: hidden; padding: 2.35rem 2.5rem;
        border: 1px solid transparent; border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(15, 11, 23, .98), rgba(20, 13, 33, .96)) padding-box,
            linear-gradient(120deg, #7c3aed, #d946ef 55%, #4c1d95) border-box;
        box-shadow: 0 24px 70px rgba(76, 29, 149, .25); margin-bottom: 1.35rem;
    }
    .hero::after {
        content: ""; position: absolute; width: 280px; height: 280px;
        right: -70px; top: -120px; border-radius: 50%;
        background: rgba(139, 92, 246, .22); filter: blur(5px);
    }
    .eyebrow {
        display: inline-flex; padding: .34rem .7rem; border-radius: 999px;
        color: #d8c9ff; background: rgba(139, 92, 246, .12);
        border: 1px solid rgba(167, 139, 250, .26); font-size: .74rem;
        font-weight: 800; letter-spacing: .11em; margin-bottom: .85rem;
    }
    .hero h1 {
        position: relative; z-index: 1; color: white; margin: 0 0 .55rem;
        font-size: clamp(2.1rem, 4vw, 3.4rem); line-height: 1.04;
        letter-spacing: -.045em;
    }
    .hero h1 span {
        background: linear-gradient(100deg, #c4b5fd, #f0abfc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p {position: relative; z-index: 1; margin: 0; color: #c9c2d4; font-size: 1.03rem;}
    [data-testid="stMetric"] {
        min-height: 116px; background: linear-gradient(145deg, #12101a, #0d0b12);
        border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 17px;
        box-shadow: inset 0 1px rgba(255, 255, 255, .025), 0 12px 30px rgba(0, 0, 0, .22);
    }
    [data-testid="stMetricLabel"] p {color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;}
    [data-testid="stMetricValue"] {color: #c4b5fd;}
    .source-card {
        color: #ddd6e8; padding: .9rem 1rem; margin: .55rem 0; border-radius: 13px;
        background: #0e0b14; border: 1px solid var(--line); border-left: 3px solid var(--violet);
    }
    .source-card strong {color: #c4b5fd;}
    .ready-badge {
        display: inline-block; padding: .38rem .72rem; border-radius: 999px;
        background: rgba(139, 92, 246, .13); color: #c4b5fd; font-weight: 750;
        border: 1px solid rgba(167, 139, 250, .28); margin-bottom: .25rem;
    }
    .step-card {
        min-height: 138px; padding: 1.15rem; border-radius: 16px;
        background: linear-gradient(145deg, #12101a, #0d0b12);
        border: 1px solid var(--line);
    }
    .step-number {color: #a78bfa; font-weight: 900; font-size: .78rem; letter-spacing: .12em;}
    .step-title {color: white; font-weight: 750; font-size: 1.04rem; margin: .5rem 0 .35rem;}
    .step-copy {color: var(--muted); font-size: .88rem; line-height: 1.5;}
    div[data-baseweb="input"] > div, [data-testid="stFileUploaderDropzone"] {
        background: #100d18; border-color: var(--line); color: var(--text);
    }
    [data-testid="stFileUploaderDropzone"]:hover {border-color: var(--violet);}
    div[data-testid="stExpander"] {
        background: rgba(16, 13, 24, .72); border: 1px solid var(--line); border-radius: 14px;
    }
    .stButton > button {
        min-height: 2.85rem; border-radius: 11px; font-weight: 750;
        color: white; border: 1px solid #44345f; background: #181320;
    }
    .stButton > button:hover {border-color: #8b5cf6; color: white; background: #21162f;}
    .stButton > button[kind="primary"] {
        border: 0; background: linear-gradient(110deg, #7c3aed, #a855f7 62%, #c026d3);
        box-shadow: 0 10px 25px rgba(124, 58, 237, .28);
    }
    .stButton > button[kind="primary"]:hover {filter: brightness(1.1);}
    [data-testid="stAlert"] {border-radius: 12px; background: #15101f; color: var(--text);}
    [data-testid="stProgress"] > div > div {background: linear-gradient(90deg, #7c3aed, #d946ef);}
    [data-testid="stChatMessage"] {
        background: rgba(17, 13, 25, .8); border: 1px solid var(--line);
        border-radius: 17px; padding: .45rem .8rem; margin-bottom: .75rem;
    }
    [data-testid="stChatInput"] {background: #100d18; border: 1px solid #35284d;}
    [data-testid="stChatInput"] textarea {color: var(--text);}
    hr {border-color: var(--line);}
    code {color: #d8b4fe; background: #171120;}
</style>
""",
    unsafe_allow_html=True,
)


def read_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_resource(show_spinner="Loading the semantic embedding model...")
def load_embedding_model():
    # Avoid an unnecessary network check when the notebook already cached the model.
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        return SentenceTransformer("all-MiniLM-L6-v2")


def initialize_state():
    defaults = {
        "processed_document": None,
        "document_hash": "",
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sources(sources: list[dict]):
    seen = set()
    for source in sources:
        identity = (source["page"], source["chunk_id"])
        if identity in seen:
            continue
        seen.add(identity)
        excerpt = source["text"].replace("\n", " ").strip()
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "..."
        excerpt = html.escape(excerpt)
        st.markdown(
            f"<div class='source-card'><strong>Page {source['page']}</strong> "
            f"· similarity {source['score']:.3f}<br>{excerpt}</div>",
            unsafe_allow_html=True,
        )


initialize_state()
configured_ocr_key = read_secret("OCR_SPACE_API_KEY")
configured_gemini_key = read_secret("GEMINI_API_KEY")
requested_model = read_secret("GEMINI_MODEL")

# Keep server-side secrets on the server. A public visitor must never receive a
# configured key as the value of a browser input, even when that input is masked.
gemini_api_key = configured_gemini_key
ocr_api_key = configured_ocr_key

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">RAG · OCR · VERIFIED SOURCES</div>
  <h1>Ask your documents.<br><span>Trust every answer.</span></h1>
  <p>Turn any PDF into a grounded conversation with page-level evidence.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
  <div class="brand-mark">D</div>
  <div><div class="brand-name">DocuMind</div><div class="brand-sub">DOCUMENT INTELLIGENCE</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("WORKSPACE")
    with st.expander("API configuration", expanded=not bool(configured_gemini_key)):
        if configured_gemini_key:
            st.success("Gemini key is configured securely")
        else:
            gemini_api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Used only for this browser session and never written to app files.",
            ).strip()

        if configured_ocr_key:
            st.success("OCR.Space key is configured securely")
        else:
            ocr_api_key = st.text_input(
                "OCR.Space API key (optional)",
                type="password",
                help="If omitted, low-text pages use local Tesseract.",
            ).strip()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.caption("Digital and scanned PDFs are supported. OCR runs only when needed.")

    if gemini_api_key:
        st.success("Gemini is configured")
    else:
        st.error("GEMINI_API_KEY is missing")

    if ocr_api_key:
        st.success("Managed OCR is configured")
    else:
        st.info("OCR.Space key missing; local Tesseract will be used")

    process_clicked = st.button(
        "Process document",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )
    if st.button("Clear document", use_container_width=True):
        st.session_state.processed_document = None
        st.session_state.document_hash = ""
        st.session_state.messages = []
        st.rerun()

if process_clicked and uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    progress_bar = st.progress(0)
    status_box = st.empty()

    def update_progress(done: int, total: int, message: str):
        denominator = max(total, 1)
        progress_bar.progress(min(done / denominator, 1.0))
        status_box.info(message)

    try:
        embedding_model = load_embedding_model()
        with st.spinner("Reading, chunking, and indexing the document..."):
            processed = process_document(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                embedding_model=embedding_model,
                ocr_space_api_key=ocr_api_key,
                progress_callback=update_progress,
            )
        st.session_state.processed_document = processed
        st.session_state.document_hash = file_hash
        st.session_state.messages = []
        progress_bar.progress(1.0)
        status_box.success("Document is ready for questions")
    except Exception as exc:
        progress_bar.empty()
        status_box.empty()
        st.error(f"Document processing failed: {exc}")

processed = st.session_state.processed_document

if processed is None:
    st.subheader("From upload to evidence in three steps")
    columns = st.columns(3)
    cards = [
        ("01 · INGEST", "Upload any PDF", "Native text is extracted first. Scanned pages automatically use OCR."),
        ("02 · INDEX", "Build document memory", "Page-aware chunks become searchable semantic vectors in FAISS."),
        ("03 · ASK", "Get grounded answers", "Gemini answers only from retrieved evidence with validated page citations."),
    ]
    for column, (number, title, copy) in zip(columns, cards):
        with column:
            st.markdown(
                f"<div class='step-card'><div class='step-number'>{number}</div>"
                f"<div class='step-title'>{title}</div><div class='step-copy'>{copy}</div></div>",
                unsafe_allow_html=True,
            )
    st.stop()

st.markdown("<span class='ready-badge'>● Document ready</span>", unsafe_allow_html=True)
st.subheader(processed.filename)

metric_columns = st.columns(4)
metric_columns[0].metric("PDF pages", processed.page_count)
metric_columns[1].metric("Extracted pages", len(processed.pages))
metric_columns[2].metric("Searchable chunks", len(processed.chunks))
metric_columns[3].metric("Vector dimensions", processed.embeddings.shape[1])

with st.expander("Extraction details"):
    st.write("Extraction methods:", processed.extraction_methods)
    st.write(f"Chunk size: 800 characters · overlap: 150 characters")
    if processed.warnings:
        for warning in processed.warnings:
            st.warning(warning)
    else:
        st.success("No extraction warnings")

st.divider()
st.subheader("Ask this document")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved evidence"):
                render_sources(message["sources"])

question = st.chat_input(
    "Ask a question about the uploaded PDF...",
    disabled=not bool(gemini_api_key),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                result = answer_question(
                    question=question,
                    processed=processed,
                    embedding_model=load_embedding_model(),
                    gemini_api_key=gemini_api_key,
                    requested_model=requested_model,
                    top_k=5,
                )
            st.markdown(result["answer"])
            st.caption(
                f"Model: {result['model']} · cited pages: "
                + ", ".join(map(str, result["cited_pages"]))
            )
            with st.expander("View retrieved evidence"):
                render_sources(result["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
