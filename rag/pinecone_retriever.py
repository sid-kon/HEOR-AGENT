"""
Pinecone retrieval for the HEOR RAG agent.

PineconeRetriever mirrors the HEORRetriever interface so agent/chain.py
needs no changes. Multi-query retrieval with cross-query deduplication
(keeping highest cosine score per chunk ID).
"""

from __future__ import annotations

import sys

# Python 3.14 compatibility shim (harmless if already applied)
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

import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "all-MiniLM-L6-v2"


class PineconeRetriever:
    """Retrieves HEOR textbook chunks from a Pinecone index."""

    def __init__(self, api_key: str, index_name: str, top_k: int = 5) -> None:
        self.top_k = top_k
        self._embedder = SentenceTransformer(_EMBED_MODEL)
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        queries: list[str],
        method_filters: list[str] = None,
        top_k_override: int = None,
    ) -> list[dict]:
        """
        Embed each query, query Pinecone, deduplicate across queries by chunk ID
        (keeping highest cosine similarity), and return the top-k results.
        """
        if not queries:
            return []

        k       = top_k_override if top_k_override is not None else self.top_k
        fetch_k = min(k * 3, 20)

        pfilter = self._build_filter(method_filters or [])

        embeddings = self._embedder.encode(queries, show_progress_bar=False)

        best: dict[str, tuple[str, dict, float]] = {}

        for q_emb in embeddings.tolist():
            kwargs: dict = {
                "vector":          q_emb,
                "top_k":           fetch_k,
                "include_metadata": True,
            }
            if pfilter:
                kwargs["filter"] = pfilter

            try:
                resp = self._index.query(**kwargs)
            except Exception:
                continue

            for match in resp.get("matches") or []:
                chunk_id = match["id"]
                score    = float(match.get("score", 0.0))
                meta     = match.get("metadata") or {}
                text     = meta.get("text", "")

                if chunk_id not in best or score > best[chunk_id][2]:
                    best[chunk_id] = (text, meta, score)

        sorted_results = sorted(best.values(), key=lambda t: t[2], reverse=True)

        return [
            {
                "text":        text,
                "source_file": meta.get("source_file", ""),
                "page_start":  meta.get("page_start", "?"),
                "score":       round(score, 4),
                "metadata":    meta,
            }
            for text, meta, score in sorted_results[:k]
        ]

    def format_context(self, results: list[dict]) -> str:
        """Serialise retrieval results into the context block for the generation prompt."""
        if not results:
            return "No relevant passages were retrieved."
        parts = [
            "[SOURCE: {}, p.{}, relevance: {:.2f}]\n{}".format(
                r["source_file"], r["page_start"], r["score"], r["text"].strip()
            )
            for r in results
        ]
        return "\n---\n".join(parts)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_filter(self, method_filters: list[str]) -> dict | None:
        """
        Build a Pinecone metadata filter for method tag filtering.
        detected_methods is stored as a list[str] in Pinecone, so we use $in.

        Single or multiple filters → {"detected_methods": {"$in": [...]}}
        Empty list → None (no filter applied)
        """
        active = [m.strip() for m in (method_filters or []) if m.strip()]
        if not active:
            return None
        return {"detected_methods": {"$in": active}}
