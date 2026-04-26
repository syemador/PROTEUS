"""
evaluation.py
=============

H.E.R.A. (Heuristic Environment for Research and Analysis) metrics for
P.R.O.T.E.U.S.

* Task 1 — Clinical QA
    - ROUGE-{1,2,L} via ``rouge-score``
    - BERTScore F1 via SciBERT (with safe context truncation)
    - Atomic-fact hallucination rate via local LLM decomposition +
      majority-vote NLI ensemble over three cross-encoders
* Task 2 — Keyword Extraction
    - Per-class and micro Precision / Recall / F1 against a gold standard
* Statistical significance
    - Paired bootstrap 95% CIs on the difference of means
    - Paired permutation tests on the sign-flipped difference vector
    - Pairwise condition report helper for Table I / II / III analysis
* Latency tracker yielding mean ± std tokens/second
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder

from llm_client import OllamaClient, resolve_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence splitter (regex-based; avoids spaCy/NLTK heavy deps)
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into trimmed sentences via a lightweight regex."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# NLI ensemble hallucination detector
# ---------------------------------------------------------------------------
#: Default three-model ensemble used in the paper.
DEFAULT_NLI_ENSEMBLE: Tuple[str, ...] = (
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-roberta-base",
)


class NliEnsembleHallucinationDetector:
    """Majority-vote ensemble over multiple NLI cross-encoders.

    Addresses the reviewer concern that single-model NLI is noisy on
    biomedical text. Each model votes entailment / not-entailment; a
    hypothesis is flagged as hallucinated unless the majority of models
    return entailment.
    """

    def __init__(
        self,
        model_names: Sequence[str] = DEFAULT_NLI_ENSEMBLE,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        requested = list(model_names)
        logger.info("Loading NLI ensemble: %s on %s", requested, self.device)

        loaded_models: List[CrossEncoder] = []
        loaded_names: List[str] = []
        failed_names: List[str] = []
        for name in requested:
            try:
                # use_fast=False added to prevent some tokenization bugs on newer transformers
                loaded_models.append(CrossEncoder(name, device=self.device, tokenizer_args={"use_fast": False}))
                loaded_names.append(name)
            except Exception as exc: 
                failed_names.append(name)
                logger.warning(
                    "Skipping NLI ensemble member %r: %s", name, exc,
                )

        if not loaded_models:
            raise RuntimeError(
                "NLI ensemble failed to load any of the requested models: "
                f"{requested}."
            )
        if len(loaded_models) < len(requested):
            logger.warning(
                "Running with a %d-model ensemble (requested %d). Majority "
                "vote thresholds adjust automatically.",
                len(loaded_models), len(requested),
            )

        self.models: List[CrossEncoder] = loaded_models
        self.model_names: List[str] = loaded_names
        self._entailment_idx: List[int] = [
            self._find_entailment_idx(m) for m in self.models
        ]

    @staticmethod
    def _find_entailment_idx(model: CrossEncoder) -> int:
        """Locate the entailment class index from the model's label map."""
        try:
            id2label = model.model.config.id2label
            for i in range(len(id2label)):
                if str(id2label[i]).lower().startswith("entail"):
                    return i
        except Exception:  
            pass
        # DeBERTa-v3 NLI default: [contradiction, entailment, neutral]
        return 1

    def predict_is_entailed(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[bool]:
        """Return ``True`` for each (premise, hypothesis) pair where the
        majority of NLI models predict entailment."""
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must be equal length")
        n = len(premises)
        if n == 0:
            return []
        pairs = list(zip(premises, hypotheses))
        votes = np.zeros((n, len(self.models)), dtype=bool)
        for m_idx, (model, ent_idx) in enumerate(
            zip(self.models, self._entailment_idx)
        ):
            scores = model.predict(
                pairs, convert_to_numpy=True, show_progress_bar=False
            )
            preds = scores.argmax(axis=1)
            votes[:, m_idx] = preds == ent_idx
        majority = votes.sum(axis=1) > (len(self.models) // 2)
        return majority.tolist()


# ---------------------------------------------------------------------------
# Atomic fact extraction
# ---------------------------------------------------------------------------
FACT_EXTRACTOR_PROMPT = """You are a fact extraction tool. Decompose the input \
text into a JSON array of atomic factual claims. Each claim must be:
  * a single self-contained statement (no conjunctions, no compound clauses)
  * asserted as factual in the text (skip questions, hedges, and meta-commentary)
  * understandable in isolation
Return ONLY a JSON array of strings. Do not include any prose."""


class AtomicFactExtractor:
    """Decomposes a generation into atomic claims via the local LLM backbone."""

    def __init__(
        self,
        llm_client: OllamaClient,
        model_alias: str = "llama3.1",
        max_tokens: int = 1024,
    ) -> None:
        self.llm = llm_client
        self.model_tag = resolve_model(model_alias)
        self.max_tokens = max_tokens

    def extract(self, text: str) -> Tuple[List[str], bool]:
        """Decompose *text* into atomic factual claims.

        Returns ``(claims, parse_failed)`` where ``parse_failed`` is ``True``
        whenever the LLM response could not be decoded as a JSON array and the
        regex sentence-splitter fallback was invoked instead.
        """
        text = (text or "").strip()
        if not text:
            return [], False
        try:
            resp = self.llm.generate(
                model=self.model_tag,
                user_prompt=text,
                system_prompt=FACT_EXTRACTOR_PROMPT,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except RuntimeError as exc:
            logger.warning(
                "Fact extractor LLM call failed (%s) — sentence-splitter fallback", exc
            )
            return split_sentences(text), True

        parsed = self._parse(resp.text)
        if parsed:
            return parsed, False

        logger.warning(
            "Fact extractor JSON parse failed (response len=%d) — sentence-splitter fallback",
            len(resp.text),
        )
        return split_sentences(text), True

    @staticmethod
    def _parse(output: str) -> List[str]:
        match = re.search(r"\[.*\]", output, re.DOTALL)
        if not match:
            return []
        try:
            arr = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(arr, list):
            return []
        return [str(x).strip() for x in arr if str(x).strip()]


# ---------------------------------------------------------------------------
# Task 1: Clinical QA metrics
# ---------------------------------------------------------------------------
@dataclass
class QaScores:
    """Per-example QA scores."""

    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bertscore_f1: float
    hallucination_rate: float
    n_sentences: int
    n_hallucinated: int
    parse_failed: bool = False  # True when JSON parse of fact-extractor response failed


class QaEvaluator:
    """Aggregate evaluator for the clinical QA task."""

    def __init__(
        self,
        device: Optional[str] = None,
        bertscore_model: str = "allenai/scibert_scivocab_uncased",
        hallucination_detector: Optional[NliEnsembleHallucinationDetector] = None,
        fact_extractor: Optional[AtomicFactExtractor] = None,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        pairs_log_path: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("QaEvaluator device=%s", self.device)
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bertscore_model = bertscore_model
        self.hallucination_detector = hallucination_detector
        self.fact_extractor = fact_extractor

        # ── Claim-chunk pair logging (Phase 1 / NLI calibration) ────────────
        # When set, every (atomic_claim, chunk, nli_decision) triple is
        # appended to this JSONL file so that N=50 NLI-unsupported pairs can
        # be sampled for manual annotation (FNR calibration of h_ctx).
        self._pairs_log_path: Optional[str] = pairs_log_path
        if self._pairs_log_path:
            # Truncate/create on init so stale runs do not accumulate.
            import pathlib
            pathlib.Path(self._pairs_log_path).write_text("", encoding="utf-8")
            logger.info(
                "Claim-chunk pair logging ENABLED → %s", self._pairs_log_path
            )

        if hallucination_detector is None:
            logger.info("Loading single NLI fallback %s", nli_model)
            self._fallback_nli: Optional[CrossEncoder] = CrossEncoder(
                nli_model, device=self.device, tokenizer_args={"use_fast": False}
            )
            self._fallback_labels: Tuple[str, ...] = (
                "contradiction",
                "entailment",
                "neutral",
            )
        else:
            self._fallback_nli = None
            self._fallback_labels = ()

    # ------------------------------------------------------------------
    def compute(
        self,
        generated: str,
        reference: str,
        retrieved_texts: List[str], # Passed as list for max-pooling
        query_id: str = "",         # Used for pair-logging (calibration)
    ) -> QaScores:
        """Score a single (generated, reference, chunks) triple."""
        rouge_scores = self.rouge.score(reference, generated)

        # TRUNCATION FIX: Aggressive 150-word limit for strict SciBERT tokenization
        safe_generated = " ".join((generated or "").split()[:150])
        safe_reference = " ".join((reference or "").split()[:150])

        _, _, f1 = bert_score(
            [safe_generated],
            [safe_reference],
            model_type=self.bertscore_model,
            device=self.device,
            verbose=False,
            rescale_with_baseline=False,
        )
        bs_f1 = float(f1[0].item())

        halluc_rate, n_units, n_halluc, parse_failed = self._hallucination(
            generated, retrieved_texts, query_id=query_id
        )
        return QaScores(
            rouge1_f=rouge_scores["rouge1"].fmeasure,
            rouge2_f=rouge_scores["rouge2"].fmeasure,
            rougeL_f=rouge_scores["rougeL"].fmeasure,
            bertscore_f1=bs_f1,
            hallucination_rate=halluc_rate,
            n_sentences=n_units,
            n_hallucinated=n_halluc,
            parse_failed=parse_failed,
        )

    # ------------------------------------------------------------------
    def _hallucination(
        self,
        generated: str,
        chunks: List[str],
        query_id: str = "",
    ) -> Tuple[float, int, int, bool]:
        """Max-pooling atomic-fact NLI hallucination rate.

        Returns ``(halluc_rate, n_units, n_hallucinated, parse_failed)``.

        When ``self._pairs_log_path`` is set every
        ``(claim, representative_chunk, nli_decision)`` triple is appended
        to that JSONL file so that NLI-unsupported pairs can be sampled for
        the manual FNR calibration audit (N=50 annotation task, TODO #1).

        For *unsupported* claims the logged chunk is the one with the highest
        ensemble vote count (most models voted entailment even if < majority),
        giving the annotator the best available evidence to judge.  For
        *supported* claims the logged chunk is the first chunk that triggered
        entailment.  When only the fallback single-model NLI is in use, the
        chunk with the highest entailment logit is logged instead.
        """
        parse_failed = False
        if self.fact_extractor is not None:
            units, parse_failed = self.fact_extractor.extract(generated)
        else:
            units = split_sentences(generated)

        if not units or not chunks:
            # Nothing generated or nothing retrieved → treat as fully hallucinated.
            return 1.0, len(units) if units else 0, len(units) if units else 0, parse_failed

        n_halluc = 0
        do_log = bool(self._pairs_log_path)

        if self.hallucination_detector is not None:
            # ── Ensemble path ────────────────────────────────────────────────
            for unit in units:
                entailed_list: List[bool] = (
                    self.hallucination_detector.predict_is_entailed(
                        chunks, [unit] * len(chunks)
                    )
                )
                supported = any(entailed_list)
                if not supported:
                    n_halluc += 1

                if do_log:
                    # For logging: pick the chunk with the MOST ensemble votes
                    # for entailment (even if below majority threshold).
                    # This gives the annotator the strongest available evidence.
                    # entailed_list is List[bool] — re-run with vote counts by
                    # querying each model separately is expensive, so we use a
                    # simpler proxy: for supported claims, pick the first
                    # entailed chunk; for unsupported, pick chunks[0] (highest
                    # retriever rank after LITM assembly).
                    if supported:
                        rep_chunk = next(
                            (c for c, e in zip(chunks, entailed_list) if e),
                            chunks[0],
                        )
                    else:
                        rep_chunk = chunks[0]
                    self._log_pair(
                        query_id=query_id,
                        claim=unit,
                        chunk=rep_chunk,
                        n_chunks=len(chunks),
                        supported=supported,
                    )

        else:
            # ── Fallback single-model path ───────────────────────────────────
            assert self._fallback_nli is not None
            for unit in units:
                pairs = [(c, unit) for c in chunks]
                scores = self._fallback_nli.predict(
                    pairs, convert_to_numpy=True, show_progress_bar=False
                )
                # scores shape: (n_chunks, n_labels)
                entail_col = list(self._fallback_labels).index("entailment") \
                    if "entailment" in self._fallback_labels else 1
                entail_scores: np.ndarray = scores[:, entail_col]
                best_chunk_idx = int(entail_scores.argmax())
                claim_supported = any(
                    self._fallback_labels[int(row.argmax())] == "entailment"
                    for row in scores
                )
                if not claim_supported:
                    n_halluc += 1

                if do_log:
                    self._log_pair(
                        query_id=query_id,
                        claim=unit,
                        chunk=chunks[best_chunk_idx],  # highest entail logit
                        n_chunks=len(chunks),
                        supported=claim_supported,
                    )

        return n_halluc / len(units), len(units), n_halluc, parse_failed

    # ------------------------------------------------------------------
    def _log_pair(
        self,
        query_id: str,
        claim: str,
        chunk: str,
        n_chunks: int,
        supported: bool,
    ) -> None:
        """Append one claim-chunk NLI decision to the pairs JSONL log.

        Each line is a self-contained JSON object with fields:
          query_id     – caller-supplied identifier for the source query
          claim        – atomic factual claim extracted from the generation
          chunk        – representative retrieved CORD-19 chunk
          n_chunks     – total number of chunks the claim was checked against
          nli_label    – "supported" | "unsupported"
          supported    – bool, True iff majority NLI voted entailment

        Only called when ``self._pairs_log_path`` is set.
        """
        if not self._pairs_log_path:
            return
        record = {
            "query_id":  query_id,
            "claim":     claim,
            "chunk":     chunk,
            "n_chunks":  n_chunks,
            "nli_label": "supported" if supported else "unsupported",
            "supported": supported,
        }
        try:
            with open(self._pairs_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write pair log: %s", exc)

    # ------------------------------------------------------------------
    def aggregate(self, per_example: Sequence[QaScores]) -> Dict[str, float]:
        """Means + 95% bootstrap CIs for every score field."""
        if not per_example:
            return {}
        fields = (
            "rouge1_f",
            "rouge2_f",
            "rougeL_f",
            "bertscore_f1",
            "hallucination_rate",
        )
        out: Dict[str, float] = {}
        for f in fields:
            vals = np.array([getattr(s, f) for s in per_example], dtype=float)
            out[f"{f}_mean"] = float(vals.mean())
            lo, hi = bootstrap_ci_single(vals)
            out[f"{f}_ci_low"] = lo
            out[f"{f}_ci_high"] = hi
        return out


# ---------------------------------------------------------------------------
# Task 2: Keyword extraction metrics
# ---------------------------------------------------------------------------
ENTITY_CLASSES: Tuple[str, ...] = ("Symptoms", "Diagnostics", "Pathogens")


@dataclass
class KeywordScores:
    """Precision / recall / F1 per entity class plus the micro average."""

    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    micro: Dict[str, float] = field(default_factory=dict)


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _prf1(predicted: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def score_keywords(
    predicted: Dict[str, Iterable[str]],
    gold: Dict[str, Iterable[str]],
) -> KeywordScores:
    """Evaluate predicted vs. gold entity dictionaries."""
    result = KeywordScores()
    all_pred: Set[str] = set()
    all_gold: Set[str] = set()

    for cls in ENTITY_CLASSES:
        p_set = {_normalize(x) for x in predicted.get(cls, []) if x}
        g_set = {_normalize(x) for x in gold.get(cls, []) if x}
        p, r, f1 = _prf1(p_set, g_set)
        result.per_class[cls] = {"precision": p, "recall": r, "f1": f1}
        all_pred.update(f"{cls}::{x}" for x in p_set)
        all_gold.update(f"{cls}::{x}" for x in g_set)

    p, r, f1 = _prf1(all_pred, all_gold)
    result.micro = {"precision": p, "recall": r, "f1": f1}
    return result


def aggregate_keyword_scores(
    per_example: Sequence[KeywordScores],
) -> Dict[str, float]:
    """Mean micro precision/recall/f1 across examples."""
    if not per_example:
        return {}
    keys = ("precision", "recall", "f1")
    out: Dict[str, float] = {}
    for k in keys:
        out[f"micro_{k}_mean"] = statistics.fmean(
            s.micro.get(k, 0.0) for s in per_example
        )
    for cls in ENTITY_CLASSES:
        for k in keys:
            out[f"{cls.lower()}_{k}_mean"] = statistics.fmean(
                s.per_class.get(cls, {}).get(k, 0.0) for s in per_example
            )
    return out


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------
@dataclass
class LatencySample:
    """A single timed generation run."""

    wall_seconds: float
    tokens: int
    tokens_per_second: float


class LatencyTracker:
    """Accumulates latency samples and computes mean/stdev tok/s."""

    def __init__(self) -> None:
        self.samples: List[LatencySample] = []

    @contextmanager
    def track(self, tokens_ref: List[int]) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            tokens = sum(tokens_ref) if tokens_ref else 0
            tps = tokens / elapsed if elapsed > 0 else 0.0
            self.samples.append(
                LatencySample(
                    wall_seconds=elapsed,
                    tokens=tokens,
                    tokens_per_second=tps,
                )
            )

    def record(self, wall_seconds: float, tokens: int) -> None:
        tps = tokens / wall_seconds if wall_seconds > 0 else 0.0
        self.samples.append(
            LatencySample(
                wall_seconds=wall_seconds, tokens=tokens, tokens_per_second=tps
            )
        )

    def summary(self) -> Dict[str, float]:
        """Return mean / stdev tok/s and mean wall time."""
        if not self.samples:
            return {
                "tokens_per_second_mean": 0.0,
                "tokens_per_second_std": 0.0,
                "wall_seconds_mean": 0.0,
                "n_samples": 0,
            }
        tps = [s.tokens_per_second for s in self.samples]
        walls = [s.wall_seconds for s in self.samples]
        return {
            "tokens_per_second_mean": statistics.fmean(tps),
            "tokens_per_second_std": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
            "wall_seconds_mean": statistics.fmean(walls),
            "n_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------
def bootstrap_ci_single(
    values: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """95% bootstrap CI on the mean of a single sample."""
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap CI on the difference ``mean(a) - mean(b)``."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if len(a) == 0:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "significant": False,
            "n": 0,
            "n_resamples": n_resamples,
        }
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1.0 - alpha))
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 0.0 or hi < 0.0),
        "n": int(n),
        "n_resamples": int(n_resamples),
    }


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-sided paired permutation test via random sign flips."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if len(a) == 0:
        return {
            "observed_diff": 0.0,
            "p_value": 1.0,
            "n": 0,
            "n_permutations": n_permutations,
        }
    diffs = a - b
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    permuted = (signs * diffs).mean(axis=1)
    extreme = int(np.sum(np.abs(permuted) >= abs(observed)))
    p = (extreme + 1) / (n_permutations + 1)
    return {
        "observed_diff": observed,
        "p_value": float(p),
        "n": int(len(diffs)),
        "n_permutations": int(n_permutations),
    }


def pairwise_condition_report(
    per_field_scores: Dict[str, Dict[str, List[float]]],
    field: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Run bootstrap + permutation across all condition pairs for one field."""
    conditions = list(per_field_scores.keys())
    report: Dict[str, Dict[str, Any]] = {}
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            ci, cj = conditions[i], conditions[j]
            a = per_field_scores[ci].get(field, [])
            b = per_field_scores[cj].get(field, [])
            if not a or not b or len(a) != len(b):
                logger.warning(
                    "Skipping %s vs %s for field %s: unpaired or empty",
                    ci,
                    cj,
                    field,
                )
                continue
            boot = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
            perm = paired_permutation_test(
                a, b, n_permutations=n_resamples, seed=seed
            )
            report[f"{ci}__vs__{cj}"] = {
                "mean_diff": boot["mean_diff"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p_value": perm["p_value"],
                "significant_ci": boot["significant"],
                "significant_p05": perm["p_value"] < 0.05,
                "n": boot["n"],
            }
    return report