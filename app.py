"""
HEOR RAG Agent — Streamlit UI
Entry point: streamlit run app.py  (run from the heor_agent/ directory)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow sibling-package imports when running from heor_agent/
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HEOR RAG Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from rag.ingestion import PDFIngestion, run_ingestion_pipeline
from rag.retriever import HEORRetriever
from agent.chain import HEORAgentChain

# ── Constants ─────────────────────────────────────────────────────────────────
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
DATA_DIR    = os.getenv("PDF_DATA_DIR", "./data")
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", "5"))

HEOR_DOMAINS = ["CEA", "BIA", "Epidemiology", "Decision Modelling", "Pharmacoecon"]


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_session_state() -> None:
    """Called once per session; subsequent reruns skip already-set keys."""
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []          # list[{role, content}]

    if "ingestion" not in st.session_state:
        st.session_state.ingestion = PDFIngestion(
            persist_dir=PERSIST_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    if "retriever" not in st.session_state:
        st.session_state.retriever = HEORRetriever(
            persist_dir=PERSIST_DIR,
            top_k=TOP_K,
        )

    if "chain" not in st.session_state:
        st.session_state.chain = None           # lazily built after API key check

    if "active_methods" not in st.session_state:
        st.session_state.active_methods = []

    if "pubmed_enabled" not in st.session_state:
        st.session_state.pubmed_enabled = True


_init_session_state()


# ── KB info helper (single collection scan per render) ────────────────────────
def _get_kb_info() -> tuple[dict, list[str], list[str]]:
    """Return (stats_dict, sorted_source_list, sorted_method_list) from one Chroma get() call."""
    try:
        ingestion: PDFIngestion = st.session_state.ingestion
        result = ingestion._collection.get(include=["metadatas"])
        metas: list[dict] = result.get("metadatas") or []
        unique_sources: set[str] = set()
        unique_methods: set[str] = set()
        for m in metas:
            if m and "source_file" in m:
                unique_sources.add(m["source_file"])
            if m and m.get("detected_methods"):
                for tag in m["detected_methods"].split(","):
                    tag = tag.strip()
                    if tag:
                        unique_methods.add(tag)
        stats = {
            "total_documents": len(metas),
            "unique_sources": len(unique_sources),
        }
        return stats, sorted(unique_sources), sorted(unique_methods)
    except Exception:
        return {"total_documents": 0, "unique_sources": 0}, [], []


def _ensure_chain() -> HEORAgentChain | None:
    """Return (and lazily create) the chain, or None if no API key is set."""
    if st.session_state.chain is not None:
        return st.session_state.chain
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        return None
    st.session_state.chain = HEORAgentChain(
        retriever=st.session_state.retriever,
        anthropic_api_key=api_key,
    )
    return st.session_state.chain


# ── Response renderer ─────────────────────────────────────────────────────────
def render_response(response_dict: dict[str, Any]) -> None:
    """
    Render a structured agent response dict inside the current st.chat_message
    block. Handles both normal responses and error dicts gracefully.
    """
    if not isinstance(response_dict, dict):
        st.error("Received an unexpected response format.")
        return

    # ── Error branch ──────────────────────────────────────────────────────────
    if "error" in response_dict:
        st.error(f"Agent error: {response_dict['error']}")
        st.caption(
            "The agent encountered an error. "
            "Check your API key and that PDFs are ingested."
        )
        if raw := response_dict.get("raw_response"):
            with st.expander("Raw response", expanded=False):
                st.code(raw, language="text")
        return

    # ── 1. Problem Diagnosis ──────────────────────────────────────────────────
    if diagnosis := response_dict.get("problem_diagnosis"):
        st.info(f"**Problem Diagnosis**\n\n{diagnosis}")

    # ── 2. Recommended method + alternatives ─────────────────────────────────
    if method := response_dict.get("recommended_method"):
        st.markdown(f"**Recommended Method:** {method}")

    alternatives = response_dict.get("alternatives_considered") or []
    if alternatives:
        with st.expander("Alternatives considered", expanded=False):
            for alt in alternatives:
                st.markdown(f"- {alt}")

    # ── 3. Identifying assumption ─────────────────────────────────────────────
    if assumption := response_dict.get("identifying_assumption"):
        st.warning(f"**Key Identifying Assumption**\n\n{assumption}")

    # ── 4. Implementation ─────────────────────────────────────────────────────
    impl = response_dict.get("implementation") or {}
    if impl:
        with st.expander("Implementation", expanded=False):
            if spec := impl.get("estimator_specification"):
                st.markdown("**Estimator Specification**")
                if "\\" in spec:
                    st.latex(spec)
                else:
                    st.code(spec, language="text")

            if code := impl.get("code_stub"):
                st.markdown("**Code Stub**")
                st.code(code, language="python")

            if params := impl.get("key_parameters"):
                st.markdown("**Key Parameters**")
                for p in params:
                    st.markdown(f"- {p}")

    # ── 5. Assumption tests ───────────────────────────────────────────────────
    tests = response_dict.get("assumption_tests") or []
    if tests:
        with st.expander("Assumption Tests", expanded=False):
            for t in tests:
                st.markdown(f"- {t}")

    # ── 6. HTA reporting ─────────────────────────────────────────────────────
    if hta := response_dict.get("hta_reporting"):
        with st.expander("HTA Reporting (CHEERS/ISPOR)", expanded=False):
            st.markdown(hta)

    # ── 7. Citations + PubMed evidence ───────────────────────────────────────
    citations = response_dict.get("citations") or []
    pubmed_context = response_dict.get("pubmed_context", "")
    has_citations = bool(citations and isinstance(citations, list))
    has_pubmed = bool(pubmed_context)

    if has_citations or has_pubmed:
        with st.expander("Citations & PubMed Evidence", expanded=True):
            if has_citations:
                rows = []
                for c in citations:
                    if isinstance(c, dict):
                        rows.append(
                            {
                                "Source": c.get("source", ""),
                                "Relevance": c.get("relevance", ""),
                            }
                        )
                if rows:
                    st.markdown("**Textbook Citations**")
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Source":    st.column_config.TextColumn(width="medium"),
                            "Relevance": st.column_config.TextColumn(width="large"),
                        },
                    )

            if has_pubmed:
                if has_citations:
                    st.divider()
                st.markdown("**PubMed Evidence Used in This Recommendation**")
                st.markdown(pubmed_context)

    # ── 8. Confidence badge ───────────────────────────────────────────────────
    confidence = response_dict.get("confidence") or {}
    if isinstance(confidence, dict):
        level     = str(confidence.get("level", "")).lower()
        rationale = confidence.get("rationale", "")
        badge_txt = f"**Confidence: {level.upper()}** — {rationale}"
        if level == "high":
            st.success(badge_txt)
        elif level == "medium":
            st.warning(badge_txt)
        elif level == "low":
            st.error(badge_txt)

    # ── 9. Retrieval debug (collapsed by default) ─────────────────────────────
    expanded_queries = response_dict.get("expanded_queries") or []
    raw_context      = response_dict.get("raw_context") or ""
    pubmed_ctx_debug = response_dict.get("pubmed_context") or ""
    with st.expander("Retrieval Debug", expanded=False):
        if expanded_queries:
            st.markdown("**Expanded Queries**")
            for q in expanded_queries:
                st.markdown(f"- `{q}`")
        else:
            st.caption("No expanded queries recorded.")

        st.markdown("**ChromaDB Context**")
        st.code(raw_context or "(no context retrieved)", language="text")

        if pubmed_ctx_debug:
            st.markdown("**PubMed Context**")
            st.code(pubmed_ctx_debug, language="text")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
stats, indexed_sources, available_methods = _get_kb_info()
total_chunks    = stats["total_documents"]
unique_sources  = stats["unique_sources"]

with st.sidebar:
    # ── 1. Header ─────────────────────────────────────────────────────────────
    st.markdown("## Knowledge Base")
    st.caption(
        f"{total_chunks:,} chunks indexed across "
        f"{unique_sources} source{'s' if unique_sources != 1 else ''}"
    )
    st.divider()

    # ── API key (show input only if not already loaded from .env) ─────────────
    env_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not env_key or env_key == "your_key_here":
        entered_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-…",
            help="Required to run the agent. Set ANTHROPIC_API_KEY in .env to skip this.",
        )
        if entered_key:
            os.environ["ANTHROPIC_API_KEY"] = entered_key
            # Force chain re-creation with new key
            st.session_state.chain = None
    else:
        st.success("API key loaded from .env ✅", icon=None)

    st.divider()

    # ── 2. PDF uploader + metadata form ──────────────────────────────────────
    st.markdown("### Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload HEOR PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "CEA reports, HTA dossiers, SLRs, budget impact models, "
            "econometrics textbooks, CSRs."
        ),
        label_visibility="collapsed",
    )

    # Metadata form (always visible so authors can pre-fill before upload)
    with st.expander("Document Metadata (optional)", expanded=bool(uploaded_files)):
        author_title = st.text_input(
            "Author / Title",
            placeholder="e.g. Wooldridge 2010 — Econometric Analysis",
        )
        chapter = st.text_input(
            "Chapter",
            placeholder="e.g. Chapter 15 — IV Estimation",
        )
        selected_domains = st.multiselect(
            "HEOR Domain",
            options=HEOR_DOMAINS,
            default=[],
            help="Tag the domain(s) covered by these PDFs.",
        )
        selected_method_tags = st.text_input(
            "Method Tags",
            placeholder="e.g. IV, PSM, Markov Model (comma-separated)",
            help="Econometric methods covered. Used for retrieval filtering.",
        )

    # ── 3. Ingest button ──────────────────────────────────────────────────────
    ingest_disabled = not uploaded_files
    if st.button(
        "Ingest PDFs",
        use_container_width=True,
        type="primary",
        disabled=ingest_disabled,
    ):
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            st.error("Set your Anthropic API key first.")
        else:
            # Save files to DATA_DIR
            saved_paths: list[str] = []
            for uf in uploaded_files:
                dest = Path(DATA_DIR) / uf.name
                dest.write_bytes(uf.read())
                saved_paths.append(str(dest))

            # Build per-file metadata overrides
            overrides: dict = {}
            if author_title:
                overrides["author"] = author_title
            if chapter:
                overrides["chapter"] = chapter
            if selected_domains:
                overrides["heor_domain"] = selected_domains  # list → comma-joined by ingestion
            if selected_method_tags:
                # Convert comma-separated string to list; ingestion joins with ", "
                overrides["detected_methods"] = [
                    t.strip() for t in selected_method_tags.split(",") if t.strip()
                ]

            metadata_list = [overrides.copy() for _ in saved_paths]

            progress = st.progress(0, text="Starting ingestion…")
            with st.spinner("Ingesting PDFs…"):
                try:
                    summaries = run_ingestion_pipeline(
                        pdf_paths=saved_paths,
                        persist_dir=PERSIST_DIR,
                        metadata_list=metadata_list if any(overrides) else None,
                    )
                    progress.progress(100, text="Done!")

                    ok   = [s for s in summaries if s["status"] == "ok"]
                    fail = [s for s in summaries if s["status"] != "ok"]
                    total_new = sum(s["total_chunks"] for s in ok)

                    if ok:
                        st.success(
                            f"Indexed **{total_new:,} chunks** from "
                            f"{len(ok)} file(s): "
                            + ", ".join(s["filename"] for s in ok)
                        )
                    for s in fail:
                        st.warning(
                            f"{s['filename']} — {s['status']}"
                        )

                    # Refresh retriever/chain so new docs are immediately queryable
                    st.session_state.retriever = HEORRetriever(
                        persist_dir=PERSIST_DIR, top_k=TOP_K
                    )
                    if st.session_state.chain is not None:
                        api_key = os.getenv("ANTHROPIC_API_KEY", "")
                        st.session_state.chain = HEORAgentChain(
                            retriever=st.session_state.retriever,
                            anthropic_api_key=api_key,
                        )

                    st.rerun()

                except Exception as exc:
                    progress.empty()
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    # ── 4. Indexed sources expander ───────────────────────────────────────────
    with st.expander(
        f"Indexed Sources ({unique_sources})",
        expanded=unique_sources > 0 and unique_sources <= 8,
    ):
        if indexed_sources:
            for src in indexed_sources:
                st.markdown(f"- `{src}`")
        else:
            st.caption("No documents ingested yet.")

    st.divider()

    # ── 5. PubMed validation toggle ───────────────────────────────────────────
    st.toggle(
        "Enable PubMed retrieval",
        key="pubmed_enabled",
        help=(
            "Before generating a response, search PubMed for peer-reviewed "
            "methodological evidence and incorporate it into the answer. "
            "Adds ~4–6 s per query."
        ),
    )

    st.divider()

    # ── 6. Clear history ──────────────────────────────────────────────────────
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chain is not None:
            st.session_state.chain.clear_history()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## HEOR Microeconometrics Agent")

chain = _ensure_chain()

# ── Method filter row ─────────────────────────────────────────────────────────
st.multiselect(
    "Active method filters",
    options=available_methods,
    key="active_methods",
    help=(
        "Restrict retrieval to chunks tagged with these methods. "
        "Tags are applied at ingestion time via Document Metadata."
    ),
)

st.divider()

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_response(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
api_ready = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
docs_ready = total_chunks > 0

if not api_ready:
    st.info("Enter your Anthropic API key in the sidebar to start querying.")
elif not docs_ready:
    st.info("Ingest at least one PDF from the sidebar to start querying.")

user_query = st.chat_input(
    "Describe your HEOR problem…",
    disabled=not (api_ready and docs_ready),
)

if user_query:
    # Display user turn immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant turn
    with st.chat_message("assistant"):
        active    = st.session_state.get("active_methods", [])
        pubmed_on = st.session_state.get("pubmed_enabled", True)
        spinner_msg = (
            "Retrieving textbook evidence + searching PubMed…"
            if pubmed_on
            else "Retrieving evidence and generating analysis…"
        )
        with st.spinner(spinner_msg):
            try:
                chain = _ensure_chain()
                if chain is None:
                    raise RuntimeError(
                        "API key is not set. Add ANTHROPIC_API_KEY to your .env file."
                    )
                result = chain.run_sync(
                    user_query,
                    active_methods=active or None,
                    pubmed_enabled=pubmed_on,
                )

                render_response(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )
            except Exception as exc:
                error_result: dict = {
                    "error": str(exc),
                }
                st.error(str(exc))
                st.caption(
                    "The agent encountered an error. "
                    "Check your API key and that PDFs are ingested."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_result}
                )
