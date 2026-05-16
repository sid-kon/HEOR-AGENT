"""
Conversational HEOR agent chain.

HEORAgentChain orchestrates the full RAG pipeline:
  1. Question rewriting (STANDALONE_QUESTION_PROMPT)
  2. Query expansion  (QUERY_EXPANSION_PROMPT)
  3. Pinecone multi-query retrieval via PineconeRetriever
  4. PubMed pre-retrieval via PubMed MCP server (optional)
  5. Enriched context = Pinecone + PubMed combined
  6. Grounded generation (RAG_GENERATION_PROMPT)
  7. Chat-history management (max 8 turns)

All Anthropic calls are made async via asyncio.to_thread so the event loop
is never blocked. run_sync() wraps run() with asyncio.run() for Streamlit.
"""

import asyncio
import re
from typing import Any, Optional

import anthropic

from rag.prompts import (
    SYSTEM_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RAG_GENERATION_PROMPT,
    FOLLOWUP_PROMPT,
    STANDALONE_QUESTION_PROMPT,
    parse_llm_json,
)

_MAX_HISTORY_TURNS = 8   # user + assistant messages combined
_MAX_RETRY = 3
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# ── PubMed MCP config ─────────────────────────────────────────────────────────
_PUBMED_MCP_SERVER: dict = {
    "type": "url",
    "url": "https://pubmed.mcp.claude.com/mcp",
    "name": "pubmed-mcp",
}

_PUBMED_SYSTEM_PROMPT = """\
You are a literature retrieval agent. Given an HEOR problem, identify the \
3 most relevant methodological approaches from PubMed. Return ONLY a JSON \
object with key 'pubmed_context' containing a single string summarising the \
most relevant methods, estimators, and published applications found. Be \
concise — maximum 400 words. Prioritise methods papers and applied \
oncology/HEOR studies from the last 10 years."""

_PUBMED_USER_TEMPLATE = """\
{standalone_question}
Detected problem type: {problem_type}
Focus on: what causal inference or survival analysis methods have been \
validated in peer-reviewed literature for this specific problem type."""


class HEORAgentChain:
    """
    Conversational agent that grounds every response in retrieved HEOR passages.

    Usage (sync, for Streamlit):
        chain = HEORAgentChain(retriever, api_key)
        result = chain.run_sync("What is the best estimator for selection bias?")

    Usage (async, for FastAPI or testing):
        result = await chain.run("What is the best estimator for selection bias?")
    """

    def __init__(
        self,
        retriever: Any,          # PineconeRetriever (or any retriever with .retrieve / .format_context)
        anthropic_api_key: str,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)

        self.chat_history: list[dict] = []   # {"role": "user"|"assistant", "content": str}
        self.last_context: str = ""          # context block from the previous turn

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        user_query: str,
        active_methods: list[str] = None,
        pubmed_enabled: bool = True,
    ) -> dict:
        """
        Execute the full RAG pipeline for one conversational turn.

        Returns a dict with all keys from RAG_GENERATION_PROMPT's JSON schema
        plus extra fields:
            raw_context      (str)  — ChromaDB context passed to the LLM
            pubmed_context   (str)  — PubMed summary used in generation (or "")
            expanded_queries (list) — queries produced by QUERY_EXPANSION_PROMPT
        """
        active_methods = active_methods or []

        # ── Step 1: rewrite question if history exists ────────────────────────
        if self.chat_history:
            standalone = await self._rewrite_question(user_query)
        else:
            standalone = user_query

        # ── Step 2: expand into multi-query + classify ────────────────────────
        expansion = await self._expand_query(standalone, active_methods)

        expanded_queries: list[str] = expansion.get("queries") or [standalone]
        problem_type: str = expansion.get("detected_problem_type", "causal_inference")
        estimand: str = expansion.get("estimand", "ATE")

        # ── Step 3: ChromaDB multi-query retrieval ────────────────────────────
        results = self.retriever.retrieve(
            queries=expanded_queries,
            method_filters=active_methods or None,
        )
        chroma_context = self.retriever.format_context(results)
        self.last_context = chroma_context

        # ── Step 4: PubMed pre-retrieval (optional) ───────────────────────────
        pubmed_context = ""
        if pubmed_enabled:
            pubmed_context = await self._fetch_pubmed_context(standalone, problem_type)

        # ── Step 5: combine ChromaDB + PubMed into enriched context ──────────
        if pubmed_context:
            enriched_context = (
                "=== TEXTBOOK METHODOLOGY (ChromaDB) ===\n"
                f"{chroma_context}\n\n"
                "=== PEER-REVIEWED EMPIRICAL EVIDENCE (PubMed) ===\n"
                f"{pubmed_context}"
            )
        else:
            enriched_context = chroma_context

        # ── Step 6: grounded generation ───────────────────────────────────────
        history_str = self._format_history(last_n=4)
        generation_prompt = RAG_GENERATION_PROMPT.format(
            retrieved_context=enriched_context,
            user_query=user_query,
            problem_type=problem_type,
            estimand=estimand,
            chat_history=history_str,
        )

        raw_answer = await self._call_claude(
            messages=[{"role": "user", "content": generation_prompt}],
            system=SYSTEM_PROMPT,
        )

        # ── Step 7: parse JSON response ───────────────────────────────────────
        parsed = parse_llm_json(raw_answer)
        if not parsed:
            parsed = {"error": "JSON parse failed", "raw_response": raw_answer}

        # ── Step 8: update chat history ───────────────────────────────────────
        self._append_turn(user_query, raw_answer)

        # ── Step 9: return enriched result ────────────────────────────────────
        return {
            **parsed,
            "raw_context": chroma_context,
            "pubmed_context": pubmed_context,
            "expanded_queries": expanded_queries,
        }

    def run_sync(
        self,
        user_query: str,
        active_methods: list[str] = None,
        pubmed_enabled: bool = True,
    ) -> dict:
        """
        Synchronous wrapper around run() for Streamlit compatibility.

        Note: will raise RuntimeError if called from within a running event
        loop (e.g. inside a Jupyter cell). In that case, use await run() directly
        or install nest_asyncio.
        """
        return asyncio.run(self.run(user_query, active_methods, pubmed_enabled))

    def clear_history(self) -> None:
        """Reset conversation memory and the stored retrieval context."""
        self.chat_history = []
        self.last_context = ""

    # ── Prompt-level helpers ──────────────────────────────────────────────────

    async def _rewrite_question(self, follow_up: str) -> str:
        """
        Call STANDALONE_QUESTION_PROMPT to make the follow-up self-contained.
        Returns the rewritten question string, falling back to the original on error.
        """
        prompt = STANDALONE_QUESTION_PROMPT.format(
            chat_history=self._format_history(last_n=4),
            follow_up_input=follow_up,
        )
        rewritten = await self._call_claude(
            messages=[{"role": "user", "content": prompt}],
        )
        return rewritten.strip() or follow_up

    async def _expand_query(
        self, retrieval_query: str, active_methods: list[str]
    ) -> dict:
        """
        Call QUERY_EXPANSION_PROMPT and parse the JSON result.
        Returns an empty dict on failure — callers must handle missing keys.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(
            user_query=retrieval_query,
            active_methods=", ".join(active_methods) if active_methods else "",
        )
        raw = await self._call_claude(
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_llm_json(raw)

    async def _fetch_pubmed_context(
        self, standalone_question: str, problem_type: str
    ) -> str:
        """
        Query PubMed via the Anthropic MCP server for methodological evidence.

        Returns a plain-text summary string (≤400 words) or an empty string
        on any failure so the caller can degrade gracefully.
        """
        user_prompt = _PUBMED_USER_TEMPLATE.format(
            standalone_question=standalone_question,
            problem_type=problem_type,
        )

        def _api_call() -> str:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=800,
                system=_PUBMED_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                mcp_servers=[_PUBMED_MCP_SERVER],  # type: ignore[arg-type]
                betas=["mcp-client-2025-04-04"],
            )
            # MCP responses may contain tool_use/tool_result blocks before
            # the final text block; extract the last text block.
            text_blocks = [
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            ]
            return text_blocks[-1] if text_blocks else ""

        try:
            raw = await asyncio.to_thread(_api_call)
        except Exception:
            return ""

        if not raw:
            return ""

        # Parse JSON envelope {"pubmed_context": "..."} if present
        parsed = parse_llm_json(raw)
        if parsed and "pubmed_context" in parsed:
            return str(parsed["pubmed_context"]).strip()

        # Fallback: return the raw text if JSON parsing fails
        return raw.strip()

    # ── Anthropic client helpers ──────────────────────────────────────────────

    async def _call_claude(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> str:
        """
        Call the Anthropic messages endpoint with exponential backoff retry
        on rate-limit errors (max 3 attempts: waits 1 s, 2 s before giving up).

        Runs the blocking SDK call in a thread so the event loop stays free.
        """
        def _api_call() -> str:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": 2000,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self._client.messages.create(**kwargs)
            return response.content[0].text

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(_MAX_RETRY):
            try:
                return await asyncio.to_thread(_api_call)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                if attempt == _MAX_RETRY - 1:
                    break
                await asyncio.sleep(2 ** attempt)   # 1 s, 2 s
            except anthropic.APIStatusError as exc:
                # Surface non-rate-limit API errors immediately.
                raise exc

        raise last_exc

    # ── History management ────────────────────────────────────────────────────

    def _append_turn(self, user_query: str, assistant_response: str) -> None:
        """
        Append a user+assistant exchange and trim to _MAX_HISTORY_TURNS total
        messages (pairs are trimmed from the oldest end).
        """
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": assistant_response})

        if len(self.chat_history) > _MAX_HISTORY_TURNS:
            # Drop oldest pair to keep history bounded.
            self.chat_history = self.chat_history[-_MAX_HISTORY_TURNS:]

    def _format_history(self, last_n: int = 4) -> str:
        """
        Serialize the last `last_n` messages as a plain-text block suitable
        for injection into prompt templates.

        Format:
            User: <content>
            Assistant: <content>
        """
        recent = self.chat_history[-last_n:] if self.chat_history else []
        if not recent:
            return "(no prior conversation)"

        lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            # Truncate very long assistant JSON blobs to 400 chars for readability.
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 400:
                content = content[:400] + " … [truncated]"
            lines.append("{}: {}".format(role_label, content))

        return "\n".join(lines)
