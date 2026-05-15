"""
Vector-store retrieval for the HEOR RAG agent.

HEORRetriever wraps a persistent Chroma collection and a SentenceTransformer
embedder, supports multi-query retrieval with cross-query deduplication, and
can filter results by detected econometric method tags.
"""

import sys
from pathlib import Path

# Python 3.14 compatibility — stub broken opentelemetry proto modules before
# chromadb is imported (same shim as rag/ingestion.py; harmless if already set).
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

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "heor_textbooks"


class HEORRetriever:
    """Retrieves HEOR textbook chunks from a persisted Chroma collection."""

    def __init__(self, persist_dir: str, top_k: int = 5) -> None:
        self.top_k = top_k
        self._embedder = SentenceTransformer(_EMBED_MODEL)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        queries: list[str],
        method_filters: list[str] = None,
        top_k_override: int = None,
    ) -> list[dict]:
        """
        Embed each query, query Chroma, deduplicate across queries by chunk ID
        (keeping the highest cosine-similarity score), and return the top-k
        results sorted by score descending.

        Args:
            queries:        One or more retrieval queries (typically from
                            QUERY_EXPANSION_PROMPT).
            method_filters: Optional list of method-tag strings (e.g. ["IV", "PSM"]).
                            Applied as OR-combined $contains filters on the
                            detected_methods metadata field.
            top_k_override: Overrides self.top_k for this call only.

        Returns:
            List of result dicts: {text, source_file, page_start, score, metadata}
        """
        if not queries:
            return []

        k = top_k_override if top_k_override is not None else self.top_k
        # Fetch more candidates per query than needed to survive deduplication.
        fetch_k = min(k * 3, 20)

        where = self._build_where_filter(method_filters or [])

        embeddings = self._embedder.encode(queries, show_progress_bar=False)
        # embeddings shape: (n_queries, embed_dim)

        query_kwargs: dict = {
            "query_embeddings": embeddings.tolist(),
            "n_results": fetch_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = self._collection.query(**query_kwargs)
        except Exception:
            # Collection may be empty or filter may match nothing — return empty.
            return []

        # Deduplicate: chunk_id -> best (text, metadata, score)
        best: dict[str, tuple[str, dict, float]] = {}
        for q_idx in range(len(queries)):
            ids = raw["ids"][q_idx]
            docs = raw["documents"][q_idx]
            metas = raw["metadatas"][q_idx]
            dists = raw["distances"][q_idx]

            for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
                score = self._score_from_distance(dist)
                if chunk_id not in best or score > best[chunk_id][2]:
                    best[chunk_id] = (text, meta, score)

        sorted_results = sorted(best.values(), key=lambda t: t[2], reverse=True)

        return [
            {
                "text": text,
                "source_file": meta.get("source_file", ""),
                "page_start": meta.get("page_start", "?"),
                "score": round(score, 4),
                "metadata": meta,
            }
            for text, meta, score in sorted_results[:k]
        ]

    def format_context(self, results: list[dict]) -> str:
        """
        Serialise retrieval results into the context block injected into
        RAG_GENERATION_PROMPT.

        Each chunk is formatted as:
            [SOURCE: {source_file}, p.{page_start}, relevance: {score:.2f}]
            {text}
            ---
        """
        if not results:
            return "No relevant passages were retrieved."

        parts = []
        for r in results:
            header = "[SOURCE: {}, p.{}, relevance: {:.2f}]".format(
                r["source_file"], r["page_start"], r["score"]
            )
            parts.append("{}\n{}".format(header, r["text"].strip()))

        return "\n---\n".join(parts)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_where_filter(self, method_filters: list[str]) -> dict | None:
        """
        Build a Chroma metadata where-clause for method tag filtering.

        Single filter  -> {"detected_methods": {"$contains": "IV"}}
        Multiple (OR)  -> {"$or": [{...}, {...}]}
        Empty list     -> None (no filter applied)
        """
        active = [m.strip() for m in (method_filters or []) if m.strip()]
        if not active:
            return None
        if len(active) == 1:
            return {"detected_methods": {"$contains": active[0]}}
        return {"$or": [{"detected_methods": {"$contains": m}} for m in active]}

    @staticmethod
    def _score_from_distance(distance: float) -> float:
        """
        Convert Chroma cosine distance to a similarity score in [0, 1].
        Chroma stores cosine distance as (1 - cosine_similarity) for the
        "cosine" HNSW space, so similarity = 1 - distance.
        Clamp to [0, 1] to guard against floating-point edge cases.
        """
        return float(np.clip(1.0 - distance, 0.0, 1.0))
