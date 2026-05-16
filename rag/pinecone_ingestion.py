"""
Pinecone ingestion pipeline for the HEOR RAG agent.

Replaces ChromaDB with Pinecone as the vector store backend.
The public interface mirrors PDFIngestion so app.py needs minimal changes.

Extraction strategy (unchanged from ingestion.py):
  1. pypdf  — fast primary extractor
  2. pdfplumber — fallback for empty pypdf pages
  3. PyMuPDF  — last resort for structurally broken PDFs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Python 3.14 compatibility — stub broken opentelemetry proto modules.
if sys.version_info >= (3, 14):
    from unittest.mock import MagicMock as _MagicMock
    _STUB_MODS = [
        "opentelemetry.proto",
        "opentelemetry.proto.common",
        "opentelemetry.proto.common.v1",
        "opentelemetry.proto.common.v1.common_pb2",
        "opentelemetry.proto.resource",
        "opentelemetry.proto.resource.v1",
        "opentelemetry.proto.resource.v1.resource_pb2",
        "opentelemetry.proto.trace",
        "opentelemetry.proto.trace.v1",
        "opentelemetry.proto.trace.v1.trace_pb2",
        "opentelemetry.proto.logs",
        "opentelemetry.proto.logs.v1",
        "opentelemetry.proto.logs.v1.logs_pb2",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.exporter.otlp.proto.grpc.exporter",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.log_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
    ]
    for _mod_name in _STUB_MODS:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = _MagicMock()

import pdfplumber
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

_EMBED_MODEL    = "all-MiniLM-L6-v2"
_EMBED_DIM      = 384
_EMBED_BATCH    = 50
_UPSERT_BATCH   = 100

# Regex patterns for content-type heuristics (identical to ingestion.py)
_NUMERIC_TOKEN_RE = re.compile(r"^-?\d[\d,.\-/%]*$")
_SYMBOL_LINE_RE   = re.compile(r"^\s*[\d\|\-=+*\\]")
_EQUATION_PATTERNS = [
    re.compile(r"=\s*[\d\w(βαγδ]"),
    re.compile(r"[βαγδθμσΩ]"),
    re.compile(r"\^|\*\*"),
    re.compile(r"[∑∏∫√]"),
    re.compile(r"\b(?:ln|log|exp|Pr|E)\s*[\[(]"),
]


def _classify_page(text: str) -> tuple[bool, bool]:
    tokens = text.split()
    lines  = [l for l in text.splitlines() if l.strip()]
    if not tokens:
        return False, False
    numeric_ratio = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t)) / len(tokens)
    symbol_ratio  = (
        sum(1 for l in lines if _SYMBOL_LINE_RE.match(l)) / len(lines) if lines else 0.0
    )
    has_table = numeric_ratio > 0.30 or symbol_ratio > 0.30
    eq_ratio  = (
        sum(1 for l in lines if any(p.search(l) for p in _EQUATION_PATTERNS)) / len(lines)
        if lines else 0.0
    )
    return has_table, eq_ratio > 0.15


class PineconeIngestion:
    """Ingests PDF documents into a Pinecone index."""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedder = SentenceTransformer(_EMBED_MODEL)

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_pdf(
        self,
        pdf_path: str,
        metadata_overrides: Optional[dict] = None,
    ) -> dict:
        """Extract, chunk, embed, and upsert a single PDF into Pinecone."""
        pdf_path = str(pdf_path)
        filename = Path(pdf_path).name
        overrides = metadata_overrides or {}

        try:
            pages = self._extract_pages(pdf_path)
        except Exception as exc:
            return {"filename": filename, "total_chunks": 0, "pages_processed": 0,
                    "status": f"extraction_failed: {exc}"}

        if not pages:
            return {"filename": filename, "total_chunks": 0, "pages_processed": 0,
                    "status": "no_text_extracted"}

        chunks, metadatas, ids = self._build_chunks(filename, pages, overrides)

        if not chunks:
            return {"filename": filename, "total_chunks": 0, "pages_processed": len(pages),
                    "status": "no_chunks_produced"}

        embeddings = self._embed_batched(chunks)
        self._upsert_batched(ids, embeddings, chunks, metadatas)

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "pages_processed": len(pages),
            "status": "ok",
        }

    def get_collection_stats(self) -> dict:
        """Return high-level statistics about the Pinecone index."""
        stats = self._index.describe_index_stats()
        total = stats.get("total_vector_count", 0)

        # List all IDs and parse unique source filenames
        unique_sources = self._list_unique_sources()

        return {
            "total_documents": total,
            "unique_sources": len(unique_sources),
            "index_name": self._index._config.host,
        }

    def get_indexed_sources_and_methods(self) -> tuple[list[str], list[str]]:
        """
        Return (sorted_source_list, sorted_method_list) by querying the index
        with diverse seed terms and collecting metadata from returned matches.

        This replaces the old list()-based approach which required 160+ paginated
        API calls (one per 100 IDs) and consistently timed out on Streamlit Cloud.
        8 targeted queries × top_k=30 = 8 API calls, reliably surfaces all sources.
        """
        # Diverse seeds that span the subject matter of all 7 indexed textbooks
        _SEED_QUERIES = [
            "cost-effectiveness analysis QALY willingness to pay threshold",
            "instrumental variables endogeneity two-stage least squares",
            "propensity score matching inverse probability weighting selection bias",
            "difference in differences parallel trends staggered adoption",
            "survival analysis censoring Kaplan-Meier Cox proportional hazards",
            "Markov model decision tree transition probability health states",
            "regression discontinuity quasi-experimental causal inference",
            "generalized linear model gamma distribution healthcare costs GLM",
        ]

        unique_sources: set[str] = set()
        unique_methods: set[str] = set()

        try:
            embeddings = self._embedder.encode(_SEED_QUERIES, show_progress_bar=False)

            for emb in embeddings.tolist():
                resp = self._index.query(
                    vector=emb,
                    top_k=30,
                    include_metadata=True,
                )
                for match in (resp.get("matches") or []):
                    meta = match.get("metadata") or {}
                    src = meta.get("source_file", "")
                    if src:
                        unique_sources.add(src)
                    for tag in (meta.get("detected_methods") or []):
                        if tag:
                            unique_methods.add(tag.strip())

        except Exception:
            pass

        return sorted(unique_sources), sorted(unique_methods)

    def delete_source(self, filename: str) -> int:
        """Delete all vectors whose source_file matches filename."""
        deleted = 0
        try:
            for id_batch in self._index.list(prefix=f"{filename}__", limit=100):
                if id_batch:
                    self._index.delete(ids=id_batch)
                    deleted += len(id_batch)
        except Exception:
            pass
        return deleted

    def _list_unique_sources(self) -> set[str]:
        sources: set[str] = set()
        try:
            for id_batch in self._index.list(limit=100):
                for vid in id_batch:
                    src = self._source_from_id(vid)
                    if src:
                        sources.add(src)
        except Exception:
            pass
        return sources

    @staticmethod
    def _source_from_id(vector_id: str) -> str:
        """Extract filename from a vector ID formatted as '{filename}__{chunk_index}'."""
        parts = vector_id.rsplit("__", 1)
        return parts[0] if len(parts) == 2 else ""

    # ── PDF extraction (identical logic to ingestion.py) ─────────────────────

    def _extract_pages(self, pdf_path: str) -> list[dict]:
        # Pass 1: pypdf
        pypdf_pages: dict[int, str] = {}
        try:
            reader = PdfReader(pdf_path)
            for n, page in enumerate(reader.pages, 1):
                pypdf_pages[n] = page.extract_text() or ""
        except Exception:
            pypdf_pages = {}

        # Pass 2: pdfplumber fallback for empty pages
        empty = {n for n, t in pypdf_pages.items() if not t.strip()}
        plumber_pages: dict[int, str] = {}
        if empty or not pypdf_pages:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    target = empty or set(range(1, len(pdf.pages) + 1))
                    for n, page in enumerate(pdf.pages, 1):
                        if n in target:
                            plumber_pages[n] = page.extract_text() or ""
            except Exception:
                pass

        # Pass 3: PyMuPDF last resort
        pymupdf_pages: dict[int, str] = {}
        if not pypdf_pages and not plumber_pages:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                for n in range(1, len(doc) + 1):
                    try:
                        pymupdf_pages[n] = doc[n - 1].get_text() or ""
                    except Exception:
                        pymupdf_pages[n] = ""
                doc.close()
            except Exception:
                pass

        total = max(
            (max(s) for s in [pypdf_pages, plumber_pages, pymupdf_pages] if s),
            default=0,
        )
        pages = []
        for n in range(1, total + 1):
            text = (
                pypdf_pages.get(n, "")
                or plumber_pages.get(n, "")
                or pymupdf_pages.get(n, "")
            )
            if not text.strip():
                continue
            has_table, has_equation = _classify_page(text)
            pages.append({"page_num": n, "text": text,
                          "has_table": has_table, "has_equation": has_equation})
        return pages

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        filename: str,
        pages: list[dict],
        overrides: dict,
    ) -> tuple[list[str], list[dict], list[str]]:
        texts: list[str]     = []
        metadatas: list[dict] = []
        ids: list[str]        = []
        chunk_index = 0

        for page in pages:
            for chunk_text in self._splitter.split_text(page["text"]):
                if not chunk_text.strip():
                    continue

                # detected_methods stored as a list for Pinecone $in filtering
                methods_raw = overrides.get("detected_methods", [])
                if isinstance(methods_raw, str):
                    methods_list = [m.strip() for m in methods_raw.split(",") if m.strip()]
                elif isinstance(methods_raw, list):
                    methods_list = [str(m).strip() for m in methods_raw if str(m).strip()]
                else:
                    methods_list = []

                meta: dict = {
                    "source_file":      filename,
                    "page_start":       page["page_num"],
                    "page_end":         page["page_num"],
                    "chunk_index":      chunk_index,
                    "has_table":        page["has_table"],
                    "has_equation":     page["has_equation"],
                    "detected_methods": methods_list,
                    "heor_domain":      "",
                    "text":             chunk_text,   # stored for fetch-based retrieval
                }

                for k, v in overrides.items():
                    if k == "detected_methods":
                        continue  # already handled above
                    meta[k] = ", ".join(v) if isinstance(v, list) else v

                # ID format: "{filename}__{chunk_index}" — double underscore separator
                ids.append(f"{filename}__{chunk_index}")
                texts.append(chunk_text)
                metadatas.append(meta)
                chunk_index += 1

        return texts, metadatas, ids

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start: start + _EMBED_BATCH]
            vecs  = self._embedder.encode(batch, show_progress_bar=False)
            all_vecs.extend(vecs.tolist())
        return all_vecs

    # ── Pinecone upsert ───────────────────────────────────────────────────────

    def _upsert_batched(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        vectors = [
            {"id": vid, "values": emb, "metadata": meta}
            for vid, emb, meta in zip(ids, embeddings, metadatas)
        ]
        for start in range(0, len(vectors), _UPSERT_BATCH):
            self._index.upsert(vectors=vectors[start: start + _UPSERT_BATCH])


# ── Module-level pipeline function ────────────────────────────────────────────

def run_pinecone_ingestion_pipeline(
    pdf_paths: list,
    api_key: str,
    index_name: str,
    metadata_list: Optional[list] = None,
) -> list[dict]:
    overrides_list = metadata_list or [None] * len(pdf_paths)
    if len(overrides_list) != len(pdf_paths):
        raise ValueError(
            f"metadata_list length ({len(overrides_list)}) must match "
            f"pdf_paths length ({len(pdf_paths)})"
        )
    ingestion = PineconeIngestion(api_key=api_key, index_name=index_name)
    return [
        ingestion.ingest_pdf(str(p), o)
        for p, o in zip(pdf_paths, overrides_list)
    ]
