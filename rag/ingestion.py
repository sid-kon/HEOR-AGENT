"""
PDF ingestion pipeline for the HEOR RAG agent.

Extraction strategy (fastest-first):
  1. pypdf  — fast, works well on most text-layer PDFs.
  2. pdfplumber — slower but more accurate on complex tables/graphics;
     used as automatic fallback if pypdf yields no text on a page.

Chunks are split with LangChain's RecursiveCharacterTextSplitter, embedded
with SentenceTransformers, and persisted to a Chroma collection.
"""

import re
import sys
from pathlib import Path
from typing import Optional

# ── Python 3.14 compatibility shim ───────────────────────────────────────────
# chromadb imports opentelemetry for telemetry, which ships proto-generated
# files that use the deprecated protobuf 3.x descriptor API
# (`_descriptor._internal_create_key`) removed in Python 3.14.
# We stub out those modules with MagicMock before chromadb is imported so the
# import chain never reaches the broken generated files.  chromadb's telemetry
# will silently no-op, which is fine for our use case.
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
import chromadb
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "heor_textbooks"
_EMBED_BATCH_SIZE = 50

# Regex patterns used in content-type heuristics
_NUMERIC_TOKEN_RE = re.compile(r"^-?\d[\d,.\-/%]*$")
_SYMBOL_LINE_RE = re.compile(r"^\s*[\d\|\-=+*\\]")
_EQUATION_PATTERNS = [
    re.compile(r"=\s*[\d\w(βαγδ]"),   # named params or assignments
    re.compile(r"[βαγδθμσΩ]"),         # Greek letters common in econometrics
    re.compile(r"\^|\*\*"),             # exponentiation
    re.compile(r"[∑∏∫√]"),             # math operators
    re.compile(r"\b(?:ln|log|exp|Pr|E)\s*[\[(]"),  # math functions
]


# ── Content-type heuristics ───────────────────────────────────────────────────

def _classify_page(text: str) -> tuple[bool, bool]:
    """
    Return (has_table, has_equation) for a single page of extracted text.

    has_table  — numeric token ratio > 30 %, or > 30 % of lines start with a
                 digit / table-border symbol.
    has_equation — > 15 % of non-empty lines contain at least one equation
                   pattern (Greek letter, operator, math function, etc.).
    """
    tokens = text.split()
    lines = [l for l in text.splitlines() if l.strip()]

    if not tokens:
        return False, False

    numeric_count = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t))
    numeric_ratio = numeric_count / len(tokens)

    symbol_line_count = sum(1 for l in lines if _SYMBOL_LINE_RE.match(l))
    symbol_ratio = symbol_line_count / len(lines) if lines else 0.0

    has_table = numeric_ratio > 0.30 or symbol_ratio > 0.30

    eq_line_count = sum(
        1 for l in lines if any(p.search(l) for p in _EQUATION_PATTERNS)
    )
    has_equation = (eq_line_count / len(lines)) > 0.15 if lines else False

    return has_table, has_equation


# ── Main class ────────────────────────────────────────────────────────────────

class PDFIngestion:
    """Ingests PDF documents into a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> None:
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._embedder = SentenceTransformer(_EMBED_MODEL)

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

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
        """
        Extract, chunk, embed, and upsert a single PDF into the collection.

        Returns a summary dict:
            {filename, total_chunks, pages_processed, status}
        """
        pdf_path = str(pdf_path)
        filename = Path(pdf_path).name
        overrides = metadata_overrides or {}

        try:
            pages = self._extract_pages(pdf_path)
        except Exception as exc:
            return {
                "filename": filename,
                "total_chunks": 0,
                "pages_processed": 0,
                "status": f"extraction_failed: {exc}",
            }

        if not pages:
            return {
                "filename": filename,
                "total_chunks": 0,
                "pages_processed": 0,
                "status": "no_text_extracted",
            }

        chunks, metadatas = self._build_chunks(filename, pages, overrides)

        if not chunks:
            return {
                "filename": filename,
                "total_chunks": 0,
                "pages_processed": len(pages),
                "status": "no_chunks_produced",
            }

        ids = ["{}_{}".format(filename, meta["chunk_index"]) for meta in metadatas]
        embeddings = self._embed_batched(chunks)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "pages_processed": len(pages),
            "status": "ok",
        }

    def get_collection_stats(self) -> dict:
        """Return high-level statistics about the current collection."""
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        unique_sources: set[str] = set()
        for meta in metadatas:
            if meta and "source_file" in meta:
                unique_sources.add(meta["source_file"])

        return {
            "total_documents": len(metadatas),
            "unique_sources": len(unique_sources),
            "collection_name": _COLLECTION_NAME,
        }

    def delete_source(self, filename: str) -> int:
        """
        Delete all chunks whose source_file metadata matches filename.
        Returns the count of deleted chunks.
        """
        result = self._collection.get(
            where={"source_file": filename},
            include=["metadatas"],
        )
        ids_to_delete = result.get("ids") or []
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_pages(self, pdf_path: str) -> list[dict]:
        """
        Extract text page-by-page, returning non-empty page dicts:
            {page_num, text, has_table, has_equation}

        Strategy (fastest-first):
          1. pypdf  — fast, handles most text-layer PDFs.
          2. pdfplumber — fallback for pages pypdf returns empty.
          3. PyMuPDF (fitz) — last resort for PDFs with structural issues
             (e.g. deep page-tree recursion) that break pypdf/pdfplumber.
        """
        # ── Pass 1: pypdf (fast) ──────────────────────────────────────────────
        pypdf_pages: dict[int, str] = {}
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pypdf_pages[page_num] = text
        except Exception:
            pypdf_pages = {}

        # ── Pass 2: pdfplumber fallback for empty pypdf pages ─────────────────
        empty_page_nums = {
            n for n, t in pypdf_pages.items() if not t.strip()
        }
        plumber_pages: dict[int, str] = {}
        if empty_page_nums or not pypdf_pages:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    target = empty_page_nums or set(range(1, len(pdf.pages) + 1))
                    for page_num, page in enumerate(pdf.pages, start=1):
                        if page_num in target:
                            plumber_pages[page_num] = page.extract_text() or ""
            except Exception:
                pass

        # ── Pass 3: PyMuPDF last resort (structurally broken PDFs) ───────────
        # Use when both pypdf and pdfplumber yield nothing at all.
        pymupdf_pages: dict[int, str] = {}
        if not pypdf_pages and not plumber_pages:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                for page_num in range(1, len(doc) + 1):
                    try:
                        text = doc[page_num - 1].get_text() or ""
                        pymupdf_pages[page_num] = text
                    except Exception:
                        pymupdf_pages[page_num] = ""
                doc.close()
            except Exception:
                pass

        # ── Merge: pypdf wins, pdfplumber fills gaps, pymupdf is last resort ──
        all_sources = [pypdf_pages, plumber_pages, pymupdf_pages]
        total_pages = max(
            (max(src) for src in all_sources if src),
            default=0,
        )
        pages = []
        for page_num in range(1, total_pages + 1):
            text = (
                pypdf_pages.get(page_num, "")
                or plumber_pages.get(page_num, "")
                or pymupdf_pages.get(page_num, "")
            )
            if not text.strip():
                continue
            has_table, has_equation = _classify_page(text)
            pages.append(
                {
                    "page_num": page_num,
                    "text": text,
                    "has_table": has_table,
                    "has_equation": has_equation,
                }
            )
        return pages

    def _build_chunks(
        self,
        filename: str,
        pages: list[dict],
        overrides: dict,
    ) -> tuple[list[str], list[dict]]:
        """
        Split each page with the text splitter, build per-chunk metadata,
        and apply caller-supplied overrides.

        Returns (texts, metadatas) in parallel lists.
        """
        texts: list[str] = []
        metadatas: list[dict] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._splitter.split_text(page["text"])
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue

                meta = {
                    "source_file": filename,
                    "page_start": page["page_num"],
                    "page_end": page["page_num"],
                    "chunk_index": chunk_index,
                    "has_table": page["has_table"],
                    "has_equation": page["has_equation"],
                    "detected_methods": "",  # comma-separated; set via metadata_overrides
                    "heor_domain": "",
                }

                # Merge overrides — caller wins on every key they supply
                for k, v in overrides.items():
                    # Chroma requires scalar metadata values
                    meta[k] = ", ".join(v) if isinstance(v, list) else v

                texts.append(chunk_text)
                metadatas.append(meta)
                chunk_index += 1

        return texts, metadatas

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in fixed-size batches; return flat list of vectors."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vecs = self._embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings


# ── Module-level pipeline function ────────────────────────────────────────────

def run_ingestion_pipeline(
    pdf_paths: list,
    persist_dir: str,
    metadata_list: Optional[list] = None,
) -> list[dict]:
    """
    Instantiate PDFIngestion and ingest each PDF in pdf_paths.

    Args:
        pdf_paths:     Ordered list of PDF file paths (str or Path).
        persist_dir:   Chroma persistence directory.
        metadata_list: Optional list of metadata override dicts, parallel to
                       pdf_paths. Pass None or an empty list to skip overrides.

    Returns:
        List of summary dicts, one per PDF, in the same order as pdf_paths.
    """
    overrides_list = metadata_list or [None] * len(pdf_paths)
    if len(overrides_list) != len(pdf_paths):
        raise ValueError(
            "metadata_list length ({}) must match pdf_paths length ({})".format(
                len(overrides_list), len(pdf_paths)
            )
        )

    ingestion = PDFIngestion(persist_dir=persist_dir)
    summaries: list[dict] = []

    for pdf_path, overrides in zip(pdf_paths, overrides_list):
        summary = ingestion.ingest_pdf(
            pdf_path=str(pdf_path),
            metadata_overrides=overrides,
        )
        summaries.append(summary)

    return summaries
