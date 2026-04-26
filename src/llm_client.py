"""
llm_client.py
=============
Lightweight Ollama REST client for P.R.O.T.E.U.S.
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, str] = {
    # ── Ollama / local ──────────────────────────────────────────────────────
    "llama3.1":         "llama3.1:8b-instruct-q4_K_M",
    "llama3.1_8b":      "llama3.1:8b-instruct-q4_K_M",
    "llama3.1_8b_q4":   "llama3.1:8b-instruct-q4_K_M",
    "llama3.1_8b_fp16": "llama3.1:8b-instruct-fp16",
    "llama3.2":         "llama3.2:1b-instruct-q4_K_M",
    "llama3.2_1b":      "llama3.2:1b-instruct-q4_K_M",
    "llama3.2_1b_q4":   "llama3.2:1b-instruct-q4_K_M",
    "llama3.2_1b_fp16": "llama3.2:1b",
    # ── Anthropic API ───────────────────────────────────────────────────────
    "claude-sonnet":    "claude-sonnet-4-6",
    "claude-opus":      "claude-opus-4-6",
    "claude-haiku":     "claude-haiku-4-5-20251001",
    # Allow full model strings to pass through as-is (both Ollama and Anthropic).
    "claude-sonnet-4-6":          "claude-sonnet-4-6",
    "claude-opus-4-6":            "claude-opus-4-6",
    "claude-haiku-4-5-20251001":  "claude-haiku-4-5-20251001",
}

def resolve_model(alias: str) -> str:
    if alias in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias]
    # Ollama raw tag (e.g. "llama3.1:8b-instruct-q4_K_M")
    if ":" in alias:
        return alias
    # Anthropic full model ID passed directly (e.g. "claude-sonnet-4-6")
    if alias.startswith("claude-"):
        return alias
    raise ValueError(f"Unknown model alias '{alias}'. Known: {sorted(MODEL_REGISTRY)}")

SYSTEM_PREAMBLE = (
    "You are P.R.O.T.E.U.S., a privacy-preserving biomedical reasoning agent. "
    "You operate inside a HIPAA-aligned local environment and must never "
    "fabricate information. Ground every statement in the retrieved context block."
)

TASK_DIRECTIVES: Dict[str, str] = {
    "qa": (
        "TASK: Clinical Question Answering.\n"
        "Generate a concise, evidence-grounded answer to the user query.\n"
        "CRITICAL INSTRUCTION: You MUST cite the provided context for EVERY claim you make. "
        "Append the citation tag (e.g., [Doc 1], [Doc 2]) at the end of the sentence it supports. "
        "If the context does not contain sufficient evidence, explicitly state 'Insufficient evidence.' "
        "Do not synthesize information from outside the provided context."
    ),
    "keywords": (
        "TASK: Medical Keyword Extraction.\n"
        "Extract medical entities from the retrieved context that are relevant to the user query, "
        "and categorize each into exactly one of three classes: Symptoms, Diagnostics, Pathogens. "
        "Return the result as JSON with keys 'Symptoms', 'Diagnostics', 'Pathogens'."
    ),
}

def build_system_prompt(task: str, context_block: str) -> str:
    if task not in TASK_DIRECTIVES:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(TASK_DIRECTIVES)}")
    return (
        f"{SYSTEM_PREAMBLE}\n\n"
        f"=== RETRIEVED CONTEXT ===\n{context_block}\n=== END CONTEXT ===\n\n"
        f"{TASK_DIRECTIVES[task]}"
    )

@dataclass
class LlmResponse:
    text: str
    model: str
    total_tokens: int
    eval_tokens: int
    prompt_tokens: int
    total_duration_s: float
    eval_duration_s: float
    tokens_per_second: float
    raw: Dict[str, Any]

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 300.0, max_retries: int = 2) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def ping(self) -> bool:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException as exc:
            logger.warning("Ollama ping failed: %s", exc)
            return False

    def list_models(self) -> List[str]:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []
        return [m.get("name", "") for m in r.json().get("models", [])]

    def generate(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 8192,
        max_tokens: Optional[int] = 1024,
        stop: Optional[List[str]] = None,
        format: Optional[Any] = None,          # NEW: JSON schema dict or "json" string
    ) -> LlmResponse:
        """Generate a completion via the Ollama /api/generate endpoint.

        Parameters
        ----------
        format:
            If provided, enables constrained structured decoding.
            Pass a JSON-Schema ``dict`` (Ollama ≥0.1.24) to enforce a specific
            output structure — e.g. the keyword extraction schema — so the sampler
            only produces tokens that keep the output a valid instance of that
            schema. Pass the string ``"json"`` for looser JSON-mode enforcement.
            Leave ``None`` (default) for unconstrained generation.

            This is the mechanism used by Experiment 1 (constrained decoding
            baseline) to bypass Format Collapse and isolate entity knowledge
            from syntactic compliance failure under 4-bit quantization.
        """
        payload: Dict[str, Any] = {
            "model": model, "prompt": user_prompt, "stream": False,
            "options": {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx},
        }
        if system_prompt is not None: payload["system"] = system_prompt
        if max_tokens is not None: payload["options"]["num_predict"] = max_tokens
        if stop: payload["options"]["stop"] = stop
        if format is not None: payload["format"] = format   # constrained decoding

        url = f"{self.host}/api/generate"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return self._to_response(r.json(), model)
            except requests.HTTPError as exc:
                logger.error("Ollama HTTP error (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
            except requests.RequestException as exc:
                logger.error("Ollama request failed (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Ollama generation failed after {self.max_retries + 1} attempts: {last_exc}")

    @staticmethod
    def _to_response(data: Dict[str, Any], model: str) -> LlmResponse:
        total_ns, eval_ns = int(data.get("total_duration", 0)), int(data.get("eval_duration", 0))
        eval_count, prompt_count = int(data.get("eval_count", 0)), int(data.get("prompt_eval_count", 0))
        eval_s = eval_ns / 1e9 if eval_ns else 0.0
        tps = (eval_count / eval_s) if eval_s > 0 else 0.0
        return LlmResponse(
            text=data.get("response", ""), model=model, total_tokens=eval_count + prompt_count,
            eval_tokens=eval_count, prompt_tokens=prompt_count, total_duration_s=total_ns / 1e9,
            eval_duration_s=eval_s, tokens_per_second=tps, raw=data,
        )


# ---------------------------------------------------------------------------
# Anthropic API client  (cloud upper-bound baseline — public-corpus only)
# ---------------------------------------------------------------------------

class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API that exposes the same
    ``generate()`` interface as ``OllamaClient`` so it can be dropped into
    the pipeline without any other changes.
    """

    def __init__(
        self,
        api_key: str,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc
        self._anthropic = _anthropic
        
        # --- PROXY REDIRECTION LOGIC ---
        import os
        jupyter_token = os.environ.get("JUPYTER_API_TOKEN")
        hub_url = os.environ.get("ANTHROPIC_BASE_URL", "https://p.l1nna.com/user/siamantar/proxy/8000")

        if jupyter_token:
            logger.info("JUPYTER_API_TOKEN detected. Routing Anthropic requests through JupyterHub proxy.")
            self._client = _anthropic.Anthropic(
                api_key="dummy-key", # Required by SDK, but ignored by your local proxy
                base_url=hub_url,
                default_headers={
                    "Authorization": f"token {jupyter_token}"
                },
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            self._client = _anthropic.Anthropic(
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
            )

    def ping(self) -> bool:
        """Best-effort liveness check: attempt a minimal API call."""
        try:
            import os
            if os.environ.get("JUPYTER_API_TOKEN"):
                return True
            self._client.models.list()
            return True
        except Exception as exc:
            logger.warning("Anthropic API ping failed: %s", exc)
            return False

    def generate(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_ctx: int = 8192,
        max_tokens: Optional[int] = 1024,
        stop: Optional[List[str]] = None,
    ) -> LlmResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if stop:
            kwargs["stop_sequences"] = stop

        import time
        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(**kwargs)
        except self._anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc
        elapsed = time.perf_counter() - t0

        text = "".join(
            block.text for block in msg.content
            if hasattr(block, "text")
        )
        prompt_tokens = msg.usage.input_tokens
        eval_tokens   = msg.usage.output_tokens
        tps = eval_tokens / elapsed if elapsed > 0 else 0.0

        return LlmResponse(
            text=text,
            model=model,
            total_tokens=prompt_tokens + eval_tokens,
            eval_tokens=eval_tokens,
            prompt_tokens=prompt_tokens,
            total_duration_s=elapsed,
            eval_duration_s=elapsed,
            tokens_per_second=tps,
            raw={"id": msg.id, "stop_reason": msg.stop_reason,
                 "usage": {"input_tokens": prompt_tokens, "output_tokens": eval_tokens}},
        )