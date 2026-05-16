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

from rag.pinecone_ingestion import PineconeIngestion, run_pinecone_ingestion_pipeline
from rag.pinecone_retriever import PineconeRetriever
from agent.chain import HEORAgentChain

# ── Constants ─────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "heor-rag")
DATA_DIR         = os.getenv("PDF_DATA_DIR", "./data")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K            = int(os.getenv("TOP_K_RETRIEVAL", "5"))

HEOR_DOMAINS = ["CEA", "BIA", "Epidemiology", "Decision Modelling", "Pharmacoecon"]


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_session_state() -> None:
    """Called once per session; subsequent reruns skip already-set keys."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "ingestion" not in st.session_state:
        if PINECONE_API_KEY:
            st.session_state.ingestion = PineconeIngestion(
                api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            st.session_state.ingestion = None

    if "retriever" not in st.session_state:
        if PINECONE_API_KEY:
            st.session_state.retriever = PineconeRetriever(
                api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX,
                top_k=TOP_K,
            )
        else:
            st.session_state.retriever = None

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "active_methods" not in st.session_state:
        st.session_state.active_methods = []

    if "pubmed_enabled" not in st.session_state:
        st.session_state.pubmed_enabled = False

    # Cache sources/methods — fetching all 16k IDs takes ~161 API calls;
    # caching avoids re-running that on every render.
    if "cached_sources" not in st.session_state:
        st.session_state.cached_sources = None   # None = not yet fetched
    if "cached_methods" not in st.session_state:
        st.session_state.cached_methods = None


_init_session_state()


# ── KB info helper ────────────────────────────────────────────────────────────
def _get_kb_info() -> tuple[dict, list[str], list[str]]:
    """
    Return (stats_dict, sorted_source_list, sorted_method_list) from Pinecone.

    Total vector count is fetched on every render (1 fast API call).
    Sources + methods are fetched once and cached in session_state — iterating
    all 16k IDs takes ~160 paginated calls which would time out on every render.
    """
    ingestion: PineconeIngestion = st.session_state.ingestion
    if ingestion is None:
        return {"total_documents": 0, "unique_sources": 0}, [], []

    # ── Fast path: total count (1 API call) ──────────────────────────────────
    total = 0
    try:
        raw_stats = ingestion._index.describe_index_stats()
        total = getattr(raw_stats, "total_vector_count", None) or raw_stats.get("total_vector_count", 0)
    except Exception:
        pass

    # ── Slow path: source + method list (cached after first fetch) ───────────
    if st.session_state.cached_sources is None:
        try:
            sources, methods = ingestion.get_indexed_sources_and_methods()
            st.session_state.cached_sources = sources
            st.session_state.cached_methods = methods
        except Exception:
            st.session_state.cached_sources = []
            st.session_state.cached_methods = []

    sources = st.session_state.cached_sources or []
    methods = st.session_state.cached_methods or []

    return {"total_documents": total, "unique_sources": len(sources)}, sources, methods


def _refresh_kb_cache() -> None:
    """Force a re-fetch of source/method list on next render."""
    st.session_state.cached_sources = None
    st.session_state.cached_methods = None


def _ensure_chain() -> HEORAgentChain | None:
    """Return (and lazily create) the chain, or None if keys are missing."""
    if st.session_state.chain is not None:
        return st.session_state.chain
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        return None
    if st.session_state.retriever is None:
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

    # ── API keys status ───────────────────────────────────────────────────────
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY not set. Add it to your .env file.")
    else:
        st.success("Pinecone connected", icon=None)

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
                    summaries = run_pinecone_ingestion_pipeline(
                        pdf_paths=saved_paths,
                        api_key=PINECONE_API_KEY,
                        index_name=PINECONE_INDEX,
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
                        st.warning(f"{s['filename']} — {s['status']}")

                    # Refresh retriever/chain so new docs are immediately queryable
                    st.session_state.retriever = PineconeRetriever(
                        api_key=PINECONE_API_KEY,
                        index_name=PINECONE_INDEX,
                        top_k=TOP_K,
                    )
                    if st.session_state.chain is not None:
                        st.session_state.chain = HEORAgentChain(
                            retriever=st.session_state.retriever,
                            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                        )
                    # Invalidate source cache so the sidebar reflects new ingestion
                    _refresh_kb_cache()

                    st.rerun()

                except Exception as exc:
                    progress.empty()
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    # ── 4. Indexed sources expander ───────────────────────────────────────────
    src_col, refresh_col = st.columns([4, 1])
    with src_col:
        src_label = f"Indexed Sources ({unique_sources})"
    with refresh_col:
        if st.button("↻", help="Refresh source list from Pinecone", key="refresh_sources"):
            _refresh_kb_cache()
            st.rerun()

    with st.expander(src_label, expanded=unique_sources > 0 and unique_sources <= 8):
        if indexed_sources:
            for src in indexed_sources:
                st.markdown(f"- `{src}`")
        else:
            st.caption("No documents ingested yet. Sources load on first render — click ↻ to refresh.")

    st.divider()

    # ── 5. PubMed validation toggle ───────────────────────────────────────────
    st.toggle(
        "PubMed evidence (higher cost)",
        key="pubmed_enabled",
        help=(
            "Search PubMed for peer-reviewed evidence before generating. "
            "Off by default — significantly increases cost and latency per query."
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
pine_ready = bool(PINECONE_API_KEY)
docs_ready = total_chunks > 0

if not pine_ready:
    st.info("Add PINECONE_API_KEY to your .env file to connect the vector store.")
elif not docs_ready:
    st.info("Ingest at least one PDF from the sidebar to start querying.")

user_query = st.chat_input(
    "Describe your HEOR problem…",
    disabled=not (pine_ready and docs_ready),
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
        with st.spinner("Generating solution…"):
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
                    "Check that PDFs are ingested and your .env keys are valid."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_result}
                )
