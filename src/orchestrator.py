"""
orchestrator.py
===============
Upgraded Pipeline tying the geometric hybrid retrieval engine to the
LLM reasoning block. Implements 'Lost-in-the-Middle' context reordering
to prevent LLM edge-bias degradation and strictly enforces citation generation.
"""

from __future__ import annotations

import json
import logging
import re
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from llm_client import (
    LlmResponse,
    build_system_prompt,
    resolve_model,
)
from retrieval import (
    InMemoryVectorStore,
    RetrievedChunk,
    format_context_block,
)

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are a strict grounding verifier for a clinical RAG system.

Given a RETRIEVED CONTEXT and a PRIMARY ANSWER, perform the following:

1. Decompose the PRIMARY ANSWER into atomic factual claims.
2. For each claim, evaluate it against the context. You MUST penalize the claim if it lacks an inline citation (e.g., [Doc X]).
3. Assign exactly one label:
   - "supported":   The claim is explicitly entailed by the cited document in the retrieved context.
   - "partial":     The claim is partially supported, or it is supported but lacks a proper [Doc X] citation.
   - "unsupported": The claim is not present in the retrieved context, contradicts it, or relies entirely on external knowledge.
4. Assign an overall verdict that is the worst label across all claims.

Return ONLY a valid JSON object matching this schema, with no prose:
{
  "claims": [{"text": "<claim>", "label": "supported|partial|unsupported"}],
  "overall": "supported|partial|unsupported"
}
"""

# Split-partial prompt: separates the two failure modes that "partial" collapses.
# Used with --verifier-mode split-partial to resolve the label-conflation confound
# described in §3.4 and §6 (RQ4) of the paper.
VERIFIER_SYSTEM_PROMPT_SPLIT = """You are a strict grounding verifier for a clinical RAG system.

Given a RETRIEVED CONTEXT and a PRIMARY ANSWER, perform the following:

1. Decompose the PRIMARY ANSWER into atomic factual claims.
2. For each claim assign exactly one label from the four below:
   - "supported":              The claim is explicitly entailed by the retrieved context AND
                               carries the required [Doc X] inline citation.
   - "weak_entailment":        The claim's content is only weakly or ambiguously entailed by
                               the retrieved context regardless of citation status.
                               This is a GROUNDING SAFETY concern.
   - "citation_noncompliance": The claim is substantively and clearly entailed by the retrieved
                               context but is missing the required [Doc X] inline citation.
                               This is a FORMATTING concern only — the content is supported.
   - "unsupported":            The claim is not present in the retrieved context, contradicts it,
                               or relies entirely on external knowledge not in the context.
3. Assign an overall verdict using the worst label observed, with severity order:
   unsupported > weak_entailment > citation_noncompliance > supported

Return ONLY a valid JSON object matching this schema, with no prose:
{
  "claims": [{"text": "<claim>", "label": "supported|weak_entailment|citation_noncompliance|unsupported"}],
  "overall": "supported|weak_entailment|citation_noncompliance|unsupported"
}
"""

# Severity ordering for overall verdict computation in split mode.
_SPLIT_SEVERITY: Dict[str, int] = {
    "supported": 0,
    "citation_noncompliance": 1,
    "weak_entailment": 2,
    "unsupported": 3,
}


@dataclass
class VerifierVerdict:
    claims: List[Dict[str, str]]
    overall: str
    n_supported: int
    n_partial: int            # standard mode only (collapsed label)
    n_unsupported: int
    raw: str
    # Split-mode counts (zero in standard mode — backward-compatible default).
    n_weak_entailment: int = 0
    n_citation_noncompliance: int = 0

    @property
    def unsupported_claim_texts(self) -> List[str]:
        return [c["text"] for c in self.claims if c.get("label") == "unsupported"]

class VerifierAgent:
    """NLI consistency check agent.

    Parameters
    ----------
    verifier_mode:
        ``"standard"``     — original three-label prompt (supported / partial / unsupported).
        ``"split-partial"`` — four-label prompt that separates ``weak_entailment``
                              (grounding safety) from ``citation_noncompliance``
                              (formatting), resolving the label-conflation confound
                              described in §3.4 and §6 RQ4 of the paper.
    """

    VALID_MODES = ("standard", "split-partial")

    def __init__(
        self,
        llm_client: Any,
        model_alias: str = "llama3.1",
        max_tokens: int = 1536,
        verifier_mode: str = "standard",
    ) -> None:
        if verifier_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown verifier_mode '{verifier_mode}'. "
                f"Choose from {self.VALID_MODES}."
            )
        self.llm = llm_client
        self.model_tag = resolve_model(model_alias)
        self.max_tokens = max_tokens
        self.verifier_mode = verifier_mode
        self._prompt = (
            VERIFIER_SYSTEM_PROMPT_SPLIT
            if verifier_mode == "split-partial"
            else VERIFIER_SYSTEM_PROMPT
        )

    def verify(self, context_block: str, primary_answer: str) -> VerifierVerdict:
        user = (
            f"=== RETRIEVED CONTEXT ===\n{context_block}\n\n"
            f"=== PRIMARY ANSWER ===\n{primary_answer}"
        )
        resp = self.llm.generate(
            model=self.model_tag, user_prompt=user, system_prompt=self._prompt,
            temperature=0.0, max_tokens=self.max_tokens,
        )
        return self._parse(resp.text, split_mode=(self.verifier_mode == "split-partial"))

    @staticmethod
    def _parse(text: str, split_mode: bool = False) -> VerifierVerdict:
        claims: List[Dict[str, str]] = []
        overall = "unsupported"
        block = _extract_json_block(text) or text

        if split_mode:
            valid_labels = {"supported", "weak_entailment", "citation_noncompliance", "unsupported"}
        else:
            valid_labels = {"supported", "partial", "unsupported"}

        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                for c in parsed.get("claims", []):
                    if not isinstance(c, dict):
                        continue
                    label = str(c.get("label", "unsupported")).lower()
                    if label not in valid_labels:
                        label = "unsupported"
                    claims.append({"text": str(c.get("text", "")).strip(), "label": label})
                overall = str(parsed.get("overall", "unsupported")).lower()
                if overall not in valid_labels:
                    overall = "unsupported"
        except json.JSONDecodeError:
            pass

        if split_mode:
            n_sup  = sum(1 for c in claims if c["label"] == "supported")
            n_we   = sum(1 for c in claims if c["label"] == "weak_entailment")
            n_cn   = sum(1 for c in claims if c["label"] == "citation_noncompliance")
            n_uns  = sum(1 for c in claims if c["label"] == "unsupported")
            # Recompute overall as worst observed label if parse gave us nothing valid.
            if claims and overall not in valid_labels:
                worst_sev = max(_SPLIT_SEVERITY.get(c["label"], 0) for c in claims)
                overall = next(k for k, v in _SPLIT_SEVERITY.items() if v == worst_sev)
            return VerifierVerdict(
                claims=claims, overall=overall,
                n_supported=n_sup, n_partial=0, n_unsupported=n_uns,
                n_weak_entailment=n_we, n_citation_noncompliance=n_cn,
                raw=text,
            )
        else:
            n_sup = sum(1 for c in claims if c["label"] == "supported")
            n_par = sum(1 for c in claims if c["label"] == "partial")
            n_uns = sum(1 for c in claims if c["label"] == "unsupported")
            if claims and overall == "unsupported" and n_uns == 0:
                overall = "partial" if n_par else "supported"
            return VerifierVerdict(
                claims=claims, overall=overall,
                n_supported=n_sup, n_partial=n_par, n_unsupported=n_uns,
                raw=text,
            )


@dataclass
class PipelineResult:
    query: str
    task: str
    model: str
    rag_enabled: bool
    retrieved: List[RetrievedChunk]
    context_block: str
    system_prompt: str
    llm_response: LlmResponse
    parsed_output: Any = None
    verifier_verdict: Optional[VerifierVerdict] = None
    regenerated: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.llm_response.text

class ProteusOrchestrator:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
    llm_client: Any,          # OllamaClient | AnthropicClient — accepts either backend
        model_alias: str = "llama3.1",
        top_k: int = 5,
        pool_k: int = 50,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        context_assembly: str = "lost-in-the-middle",
        redundancy_threshold: float = 0.85,
        rag_enabled: bool = True,
        verifier: Optional[VerifierAgent] = None,
        regenerate_on_unsupported: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm_client
        self.model_alias = model_alias
        self.model_tag = resolve_model(model_alias)
        self.top_k = top_k
        self.pool_k = pool_k
        self.retrieval_mode = retrieval_mode
        self.context_assembly = context_assembly
        self.redundancy_threshold = redundancy_threshold
        self.rag_enabled = rag_enabled
        self.verifier = verifier
        self.regenerate_on_unsupported = regenerate_on_unsupported

        self.reranker = None
        if use_reranker and self.rag_enabled:
            try:
                from sentence_transformers import CrossEncoder
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading CrossEncoder Reranker on %s...", device)
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            except Exception as e:
                logger.warning("Failed to load reranker: %s. Falling back to dense scoring.", e)

    def _apply_lost_in_the_middle(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Reorders chunks: Best at start, second best at end, weaker in the middle."""
        if not chunks: return chunks
        result = [None] * len(chunks)
        left, right = 0, len(chunks) - 1
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                result[left] = chunk
                left += 1
            else:
                result[right] = chunk
                right -= 1
        return result

    def run(self, query: str, task: str = "qa", temperature: float = 0.2, max_tokens: int = 1024) -> PipelineResult:
        logger.info("Pipeline | mode=%s reranker=%s | model=%s", self.retrieval_mode, self.reranker is not None, self.model_tag)

        retrieved: List[RetrievedChunk] = []
        if self.rag_enabled:
            retrieved = self.vector_store.search(
                query, top_k=self.top_k, pool_k=self.pool_k, retrieval_mode=self.retrieval_mode,
                reranker=self.reranker, redundancy_threshold=self.redundancy_threshold,
            )
            if self.context_assembly == "lost-in-the-middle":
                retrieved = self._apply_lost_in_the_middle(retrieved)

        context_block = format_context_block(retrieved) if self.rag_enabled else "(retrieval disabled)"
        system_prompt = build_system_prompt(task=task, context_block=context_block)

        llm_response = self.llm.generate(
            model=self.model_tag, user_prompt=query, system_prompt=system_prompt, 
            temperature=temperature, max_tokens=max_tokens,
        )

        verdict: Optional[VerifierVerdict] = None
        regenerated = False
        if self.verifier is not None and task == "qa" and self.rag_enabled:
            verdict = self.verifier.verify(context_block, llm_response.text)
            if self.regenerate_on_unsupported and verdict.n_unsupported > 0:
                avoid_list = "\n".join(f"- {c}" for c in verdict.unsupported_claim_texts)
                augmented = system_prompt + "\n\n=== AVOID UNSUPPORTED CLAIMS ===\n" + avoid_list
                llm_response = self.llm.generate(
                    model=self.model_tag, user_prompt=query, system_prompt=augmented, 
                    temperature=temperature, max_tokens=max_tokens,
                )
                verdict = self.verifier.verify(context_block, llm_response.text)
                regenerated = True

        parsed: Any = _parse_keyword_json(llm_response.text) if task == "keywords" else None

        return PipelineResult(
            query=query, task=task, model=self.model_tag, rag_enabled=self.rag_enabled,
            retrieved=retrieved, context_block=context_block, system_prompt=system_prompt,
            llm_response=llm_response, parsed_output=parsed, verifier_verdict=verdict,
            regenerated=regenerated, meta={"top_k": self.top_k, "mode": self.retrieval_mode}
        )

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
def _extract_json_block(text: str) -> Optional[str]:
    m = _JSON_BLOCK_RE.search(text)
    return m.group(0) if m else None

def _parse_keyword_json(text: str) -> Dict[str, List[str]]:
    if not text: return {"Symptoms": [], "Diagnostics": [], "Pathogens": []}
    for candidate in (text, _extract_json_block(text)):
        if not candidate: continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {
                    "Symptoms": _as_str_list(parsed.get("Symptoms", [])),
                    "Diagnostics": _as_str_list(parsed.get("Diagnostics", [])),
                    "Pathogens": _as_str_list(parsed.get("Pathogens", [])),
                }
        except json.JSONDecodeError: continue
    return {"Symptoms": [], "Diagnostics": [], "Pathogens": []}

def _as_str_list(x: Any) -> List[str]:
    if isinstance(x, list): return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str): return [v.strip() for v in x.split(",") if v.strip()]
    return []