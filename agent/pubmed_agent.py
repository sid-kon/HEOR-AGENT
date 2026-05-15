"""
PubMed Validation Agent for the HEOR RAG pipeline.

PubMedValidationAgent performs a second-pass literature validation against
PubMed using the Anthropic API with the PubMed MCP server. It takes the
output dict from HEORAgentChain.run_sync() and enriches it with:

    pubmed_validation: {
        foundational_reference: str,
        applied_studies: list[dict],
        critiques: str | list,
        overall_evidence_grade: "A" | "B" | "C",
        pubmed_ids: list[str],
    }
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2000
_MCP_SERVER: dict[str, str] = {
    "type": "url",
    "url": "https://pubmed.mcp.claude.com/mcp",
    "name": "pubmed-mcp",
}

_SYSTEM_PROMPT = """\
You are a systematic literature validation agent for HEOR methodology.
Given a recommended econometric method and clinical context, search PubMed
for empirical studies that either validate or challenge that method in
similar disease areas. Always retrieve: (a) the most cited methodological
paper establishing the approach, (b) 2-3 recent applied studies using it
in comparable disease/data contexts, (c) any published critiques or
simulation studies showing limitations. Return results as JSON with keys:
foundational_reference, applied_studies, critiques, overall_evidence_grade
(A/B/C based on consistency of findings), pubmed_ids."""

_USER_PROMPT_TEMPLATE = """\
Recommended method: {recommended_method}
Clinical context: {disease_area}
Estimand: {estimand}
Key assumption being made: {identifying_assumption}
Search PubMed for empirical validation of this methodological approach
in this clinical context. Focus on studies using real-world claims or
registry data published in the last 8 years."""

# Disease/context patterns for heuristic extraction from problem_diagnosis
_DISEASE_PATTERNS: list[str] = [
    r"\b(?:diabetes|diabetic|GLP-1|insulin|glycemic)\b",
    r"\b(?:cardiovascular|cardiac|heart failure|coronary|MACE|stroke)\b",
    r"\b(?:cancer|oncology|tumor|carcinoma|neoplasm|chemotherapy)\b",
    r"\b(?:respiratory|COPD|asthma|pulmonary)\b",
    r"\b(?:hypertension|blood pressure|antihypertensive)\b",
    r"\b(?:mental health|depression|anxiety|psychiatric|schizophrenia)\b",
    r"\b(?:HIV|AIDS|infectious disease)\b",
    r"\b(?:rheumatoid arthritis|autoimmune|biologic)\b",
    r"\b(?:renal|kidney|CKD|nephro)\b",
    r"\b(?:neurological|Alzheimer|Parkinson|dementia)\b",
    r"\b(?:Medicare|Medicaid|claims|EHR|registry|RWE|real.world)\b",
]


class PubMedValidationAgent:
    """
    Second-pass PubMed literature validation for HEORAgentChain responses.

    Usage:
        validator = PubMedValidationAgent(api_key)
        enriched = validator.validate(heor_response)
    """

    def __init__(self, anthropic_api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, heor_response: dict) -> dict:
        """
        Run PubMed validation on a HEORAgentChain output dict.

        Args:
            heor_response: The dict returned by HEORAgentChain.run_sync().

        Returns:
            The same dict with a 'pubmed_validation' key merged in.
            On any error the key still appears with an 'error' sub-key so the
            UI can render a graceful fallback.
        """
        if "error" in heor_response:
            return heor_response

        try:
            recommended_method = heor_response.get("recommended_method", "").strip()
            if not recommended_method:
                return heor_response

            problem_diagnosis   = heor_response.get("problem_diagnosis", "")
            identifying_assumption = heor_response.get("identifying_assumption", "")
            estimand = self._extract_estimand(heor_response)
            disease_area = self._parse_disease_area(problem_diagnosis)

            user_prompt = _USER_PROMPT_TEMPLATE.format(
                recommended_method=recommended_method,
                disease_area=disease_area,
                estimand=estimand,
                identifying_assumption=identifying_assumption,
            )

            raw = self._call_pubmed_agent(user_prompt)
            validation = self._parse_response(raw)

        except Exception as exc:
            validation = {
                "error": str(exc),
                "overall_evidence_grade": "N/A",
            }

        return {**heor_response, "pubmed_validation": validation}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_disease_area(self, problem_diagnosis: str) -> str:
        """
        Extract disease/clinical context from problem_diagnosis via regex.
        Falls back to the first sentence (truncated) or a generic label.
        """
        if not problem_diagnosis:
            return "healthcare and clinical outcomes"

        found: list[str] = []
        for pattern in _DISEASE_PATTERNS:
            m = re.search(pattern, problem_diagnosis, re.IGNORECASE)
            if m:
                found.append(m.group())

        if found:
            # Deduplicate while preserving insertion order
            return ", ".join(dict.fromkeys(found))

        first_sentence = problem_diagnosis.split(".")[0].strip()
        return first_sentence[:150] if first_sentence else "healthcare and clinical outcomes"

    def _extract_estimand(self, heor_response: dict) -> str:
        """
        Pull estimand from the response; fall back through several fields.
        """
        # Direct key (not in schema but sometimes present)
        if est := heor_response.get("estimand", "").strip():
            return est
        # Expanded queries may mention it
        for q in heor_response.get("expanded_queries") or []:
            for term in ("ATE", "ATT", "LATE", "CEA", "CUA", "BIA", "NMB"):
                if term in q:
                    return term
        return "ATE"

    def _call_pubmed_agent(self, user_prompt: str) -> str:
        """
        Send the validation query to Claude via the PubMed MCP server.

        The response content may contain MCP tool_use / tool_result blocks
        before the final text block; we extract the last text block.
        """
        response = self._client.beta.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            mcp_servers=[_MCP_SERVER],  # type: ignore[arg-type]
            betas=["mcp-client-2025-04-04"],
        )

        # Collect all text blocks (MCP may prepend tool_use/tool_result blocks)
        text_blocks = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        ]
        return text_blocks[-1] if text_blocks else ""

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """
        Strip markdown fences, extract the first JSON object, and normalise
        the expected keys into a consistent schema.
        """
        if not raw:
            return {"error": "Empty response from PubMed agent", "overall_evidence_grade": "N/A"}

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return {"raw_text": raw[:500], "overall_evidence_grade": "N/A"}

        try:
            data: dict = json.loads(m.group())
        except json.JSONDecodeError:
            return {"raw_text": raw[:500], "overall_evidence_grade": "N/A"}

        # ── Normalise applied_studies → list[dict] ────────────────────────────
        applied: list[Any] = data.get("applied_studies") or []
        normalised: list[dict] = []
        for s in applied:
            if isinstance(s, dict):
                normalised.append(s)
            elif isinstance(s, str):
                normalised.append({"title": s})
        data["applied_studies"] = normalised

        # ── Normalise pubmed_ids → list[str] ──────────────────────────────────
        pmids = data.get("pubmed_ids") or []
        if isinstance(pmids, str):
            data["pubmed_ids"] = [p.strip() for p in pmids.split(",") if p.strip()]

        # ── Ensure grade is uppercase single letter ────────────────────────────
        grade = str(data.get("overall_evidence_grade", "N/A")).strip().upper()
        data["overall_evidence_grade"] = grade if grade in ("A", "B", "C") else "N/A"

        return data
