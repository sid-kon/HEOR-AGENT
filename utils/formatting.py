import re
from typing import List, Optional

import pandas as pd
import streamlit as st
from langchain.schema import Document


def format_source_citations(docs: List[Document]) -> List[dict]:
    """Convert retrieved Document objects into citation dicts for display."""
    citations = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "Document": source,
                "Page": page,
                "Excerpt": doc.page_content[:300].strip() + "…",
            }
        )
    return citations


def build_sources_dataframe(docs: List[Document]) -> pd.DataFrame:
    citations = format_source_citations(docs)
    if not citations:
        return pd.DataFrame(columns=["Document", "Page", "Excerpt"])
    return pd.DataFrame(citations)


def extract_confidence_score(answer: str) -> Optional[float]:
    """
    Parse an explicit confidence token the LLM may include, e.g.:
      [CONFIDENCE: 0.82]
    Returns a float in [0, 1] or None if not found.
    """
    match = re.search(r"\[CONFIDENCE:\s*([\d.]+)\]", answer, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def strip_confidence_token(answer: str) -> str:
    return re.sub(r"\[CONFIDENCE:\s*[\d.]+\]", "", answer, flags=re.IGNORECASE).strip()


def render_answer_with_sources(
    answer: str,
    source_docs: List[Document],
    show_sources: bool = True,
) -> None:
    """Render the LLM answer and collapsible source citations in Streamlit."""
    confidence = extract_confidence_score(answer)
    clean_answer = strip_confidence_token(answer)

    st.markdown(clean_answer)

    if confidence is not None:
        color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
        st.markdown(
            f"<small>Evidence confidence: "
            f"<span style='color:{color};font-weight:bold'>{confidence:.0%}</span></small>",
            unsafe_allow_html=True,
        )

    if show_sources and source_docs:
        with st.expander(f"Sources ({len(set((d.metadata.get('source'), d.metadata.get('page')) for d in source_docs))} passages)", expanded=False):
            df = build_sources_dataframe(source_docs)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Excerpt": st.column_config.TextColumn(width="large"),
                },
            )


def render_ingestion_summary(num_chunks: int, sources: List[dict]) -> None:
    """Display a compact ingestion summary card."""
    col1, col2 = st.columns(2)
    col1.metric("Chunks indexed", num_chunks)
    col2.metric("Documents", len(sources))
    if sources:
        df = pd.DataFrame(sources)
        st.dataframe(df, use_container_width=True, hide_index=True)


def chat_message_html(role: str, content: str) -> str:
    """Return simple HTML for a chat bubble (fallback when st.chat_message unavailable)."""
    align = "right" if role == "user" else "left"
    bg = "#DCF8C6" if role == "user" else "#F1F0F0"
    return (
        f"<div style='text-align:{align};margin:4px 0'>"
        f"<span style='background:{bg};padding:8px 12px;border-radius:12px;"
        f"display:inline-block;max-width:80%;text-align:left'>{content}</span></div>"
    )
