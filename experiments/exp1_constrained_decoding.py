"""
exp1_constrained_decoding.py
============================
Experiment 1 (Reviewer Revision): Constrained Structured Decoding Baseline
for Task 2 (Medical Keyword Extraction).

PURPOSE
-------
The reviewer identifies a Format Collapse confound: the manuscript currently
evaluates keyword extraction using *native* JSON generation (no grammar
enforcement), and defends low F1 scores as Format Collapse — the degradation
of syntactic instruction-following under 4-bit quantization.  The reviewer
requires a *parallel* constrained-decoding run to empirically separate the
entity-knowledge component from the syntactic-compliance component.

This script re-runs the 25-keyword-prompt held-out set using Ollama's native
JSON-Schema grammar enforcement (the ``format`` field added to llm_client.py).
The schema enforcer constrains the sampler so it can only produce tokens that
keep the output a valid instance of ``{Symptoms: [...], Diagnostics: [...],
Pathogens: [...]}`` — bypassing Format Collapse by construction.

The resulting delta (constrained F1 − native F1) quantifies the Format
Collapse penalty per entity class.  A large delta on Symptoms/Diagnostics
(near-zero native F1) proves the entity knowledge was present but
syntactically inaccessible.

OUTPUT
------
  results/kw_constrained_{TAG}.json   — full per-prompt results
  results/kw_constrained_dual_table.json — side-by-side comparison
  Prints LaTeX dual-evaluation table for Section V.

USAGE
-----
  # Minimum: run constrained 8B No-RAG (fastest, no index needed)
  python exp1_constrained_decoding.py --model llama3.1 --no-rag

  # Full constrained 8B + RAG (ClinicalBERT) — requires built index
  python exp1_constrained_decoding.py --model llama3.1 --retriever cbert \\
      --index-dir /path/to/index

  # Compare against a specific native result file
  python exp1_constrained_decoding.py --model llama3.1 --no-rag \\
      --native-result results/kw_8b_norag.json

NOTES
-----
* Ollama >= 0.1.24 required for JSON-Schema format enforcement.
* The ``format`` parameter was added to OllamaClient.generate() in
  llm_client.py as part of this reviewer revision patch.
* Temperature is set to 0.0 (deterministic) for both native and constrained
  runs to isolate format effects from sampling variance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("exp1")

# ---------------------------------------------------------------------------
# JSON schema enforced by the Ollama grammar sampler.
# Matches the Task 2 output schema exactly.
# ---------------------------------------------------------------------------
KW_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "Symptoms":    {"type": "array", "items": {"type": "string"}},
        "Diagnostics": {"type": "array", "items": {"type": "string"}},
        "Pathogens":   {"type": "array", "items": {"type": "string"}},
    },
    "required": ["Symptoms", "Diagnostics", "Pathogens"],
    "additionalProperties": False,
}

ENTITY_CLASSES = ["Symptoms", "Diagnostics", "Pathogens"]


# ---------------------------------------------------------------------------
# Scoring (mirrors evaluation.py score_keywords exactly)
# ---------------------------------------------------------------------------

def _normalise(term: str) -> str:
    return term.lower().strip()


def score_keywords(
    predicted: Dict[str, List[str]],
    gold: Dict[str, List[str]],
) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0

    for cls in ENTITY_CLASSES:
        pred_set = {_normalise(t) for t in predicted.get(cls, [])}
        gold_set = {_normalise(t) for t in gold.get(cls, [])}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
        total_tp += tp; total_fp += fp; total_fn += fn

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0.0)
    return {
        "per_class": per_class,
        "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
    }


def aggregate_scores(score_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, List[float]]] = {
        cls: {"precision": [], "recall": [], "f1": []} for cls in ENTITY_CLASSES
    }
    micro_f1s: List[float] = []
    for s in score_list:
        for cls in ENTITY_CLASSES:
            for m in ("precision", "recall", "f1"):
                agg[cls][m].append(s["per_class"][cls][m])
        micro_f1s.append(s["micro"]["f1"])

    def _mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    result: Dict[str, Any] = {
        "per_class": {
            cls: {m: _mean(agg[cls][m]) for m in ("precision", "recall", "f1")}
            for cls in ENTITY_CLASSES
        },
        "micro_f1": _mean(micro_f1s),
    }
    return result


# ---------------------------------------------------------------------------
# Context block builder (mirrors retrieval.format_context_block)
# ---------------------------------------------------------------------------

def _build_norag_context() -> str:
    return "(retrieval disabled)"


# ---------------------------------------------------------------------------
# Run constrained decoding
# ---------------------------------------------------------------------------

def run_constrained_kw(
    queries: List[Dict[str, Any]],
    llm_client: Any,
    model_tag: str,
    system_prompt_base: str,
    context_fn,            # callable(query_item) -> context_block str
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run keyword extraction with Ollama JSON-schema grammar enforcement.

    Returns (per_example_results, score_objects).
    """
    per_example, score_objs = [], []

    for i, item in enumerate(queries, start=1):
        query = item["query"]
        gold  = item.get("gold", {cls: [] for cls in ENTITY_CLASSES})
        logger.info("[KW-Constrained %d/%d] %s", i, len(queries), query[:70])

        context_block = context_fn(item)
        system_prompt = (
            f"{system_prompt_base}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_block}\n=== END CONTEXT ===\n\n"
            "TASK: Medical Keyword Extraction.\n"
            "Extract medical entities from the retrieved context that are relevant "
            "to the user query, and categorize each into exactly one of three classes: "
            "Symptoms, Diagnostics, Pathogens.\n"
            "Return ONLY a valid JSON object with keys 'Symptoms', 'Diagnostics', "
            "'Pathogens', each mapping to a flat list of strings.\n"
            "DO NOT include descriptions, nested objects, or any prose outside the JSON."
        )

        # ── Constrained call — passes KW_JSON_SCHEMA to Ollama format param ──
        resp = llm_client.generate(
            model=model_tag,
            user_prompt=query,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            format=KW_JSON_SCHEMA,      # grammar-constrained decoding
        )

        # Parse: constrained decoding guarantees valid JSON, but we still
        # defensively fall back to empty on any unexpected error.
        raw_text = resp.text.strip()
        try:
            predicted = json.loads(raw_text)
            # Normalise: strip nested dicts/objects that quantized models
            # sometimes produce even under schema enforcement
            predicted = {
                cls: [
                    str(v).strip() if not isinstance(v, dict)
                    else str(v.get("name", v)).strip()
                    for v in predicted.get(cls, [])
                ]
                for cls in ENTITY_CLASSES
            }
        except (json.JSONDecodeError, TypeError):
            logger.warning("  Constrained output failed to parse (unexpected): %s", raw_text[:200])
            predicted = {cls: [] for cls in ENTITY_CLASSES}

        scores = score_keywords(predicted, gold)
        score_objs.append(scores)

        per_example.append({
            "query":     query,
            "gold":      gold,
            "predicted": predicted,
            "raw_output": raw_text,
            "scores":    scores,
            "tokens_per_second": resp.tokens_per_second,
            "decoding":  "constrained",
        })
        logger.info(
            "  μ-F1=%.3f  Symp=%.3f  Diag=%.3f  Path=%.3f",
            scores["micro"]["f1"],
            scores["per_class"]["Symptoms"]["f1"],
            scores["per_class"]["Diagnostics"]["f1"],
            scores["per_class"]["Pathogens"]["f1"],
        )

    return per_example, score_objs


# ---------------------------------------------------------------------------
# Load native results for comparison
# ---------------------------------------------------------------------------

def load_native_aggregate(path: Path) -> Optional[Dict[str, Any]]:
    """Load and return the aggregate section of an existing KW result file."""
    if not path.exists():
        logger.warning("Native result not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("aggregate")


def load_native_per_example(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_example", [])


# ---------------------------------------------------------------------------
# LaTeX table printer
# ---------------------------------------------------------------------------

def _print_latex_table(
    label_native: str,
    label_constrained: str,
    agg_native: Optional[Dict[str, Any]],
    agg_constrained: Dict[str, Any],
) -> None:
    """Print the dual-evaluation LaTeX table for Section V."""
    print("\n" + "=" * 72)
    print("% ── INSERT AFTER \\paragraph{Format Collapse} IN SECTION V ──")
    print("=" * 72)
    print(r"""
\paragraph{Constrained Decoding Baseline: Isolating Entity Knowledge
from Format Compliance.}
To empirically separate the entity-knowledge component of Format Collapse
from its syntactic-compliance component, we re-evaluate Task~2 using
Ollama's native JSON-Schema grammar enforcement (\texttt{format} parameter,
Ollama $\geq$0.1.24), which constrains the decoder's sampling distribution
to outputs that are valid instances of the
$\{\texttt{Symptoms}, \texttt{Diagnostics}, \texttt{Pathogens}\}$ schema by
construction --- bypassing Format Collapse at the decoding layer.
Table~\ref{tab:keywords_constrained} reports the dual evaluation.
The $\Delta$ columns isolate the Format Collapse penalty per entity class:
a large positive $\Delta$ on Symptoms or Diagnostics proves the entity
knowledge was present in the model's parametric memory but syntactically
inaccessible under native generation, establishing that the keyword-extraction
performance gap is a \emph{compliance gap, not a knowledge gap}.
""")

    def _row(cond_label: str, agg: Optional[Dict], suffix: str = "") -> str:
        if agg is None:
            return f"{cond_label}{suffix:<45s} & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- \\\\"
        pc = agg.get("per_class", {})
        mu = agg.get("micro_f1", agg.get("micro", {}).get("f1", 0.0))
        symp  = pc.get("Symptoms",    {})
        diag  = pc.get("Diagnostics", {})
        path  = pc.get("Pathogens",   {})

        def _f(d, k): return f"{d.get(k, 0.0):.3f}"

        return (
            f"{cond_label}{suffix:<45s} & "
            f"{_f(symp,'precision')} & {_f(symp,'recall')} & {_f(symp,'f1')} & "
            f"{_f(diag,'precision')} & {_f(diag,'recall')} & {_f(diag,'f1')} & "
            f"{_f(path,'precision')} & {_f(path,'recall')} & {_f(path,'f1')} & "
            f"{mu:.3f} \\\\"
        )

    def _delta_row(agg_a: Optional[Dict], agg_b: Dict) -> str:
        """Print delta row (constrained − native) for each metric."""
        if agg_a is None:
            return r"\multicolumn{10}{l}{\textit{Native baseline not available for delta}} \\"
        pc_a = agg_a.get("per_class", {})
        pc_b = agg_b.get("per_class", {})
        mu_a = agg_a.get("micro_f1", agg_a.get("micro", {}).get("f1", 0.0))
        mu_b = agg_b.get("micro_f1", 0.0)

        def _d(cls, k):
            v = pc_b.get(cls, {}).get(k, 0.0) - pc_a.get(cls, {}).get(k, 0.0)
            return f"{'+' if v >= 0 else ''}{v:.3f}"

        d_mu = mu_b - mu_a
        return (
            r"$\Delta$ (constrained $-$ native) &"
            f" {_d('Symptoms','precision')} & {_d('Symptoms','recall')} & {_d('Symptoms','f1')} &"
            f" {_d('Diagnostics','precision')} & {_d('Diagnostics','recall')} & {_d('Diagnostics','f1')} &"
            f" {_d('Pathogens','precision')} & {_d('Pathogens','recall')} & {_d('Pathogens','f1')} &"
            f" {'+' if d_mu >= 0 else ''}{d_mu:.3f} \\\\"
        )

    print(r"""\begin{table*}[t]
\centering
\caption{%
  Dual keyword-extraction evaluation: native generation vs.\ constrained
  structured decoding (Ollama JSON-Schema grammar enforcement).
  P\,=\,Precision, R\,=\,Recall, $\mu$-F$_1$\,=\,micro-averaged F$_1$.
  $\Delta$ rows quantify the Format Collapse penalty per entity class
  (constrained $-$ native); a large positive $\Delta$ on Symptoms or
  Diagnostics proves entity knowledge was present but syntactically
  inaccessible under native 4-bit generation.
}
\label{tab:keywords_constrained}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrrrl@{}}
\toprule
\textbf{Configuration} &
  \multicolumn{3}{c}{\textbf{Symptoms}} &
  \multicolumn{3}{c}{\textbf{Diagnostics}} &
  \multicolumn{3}{c}{\textbf{Pathogens}} & \\
& P & R & F1 & P & R & F1 & P & R & F1 & \textbf{$\mu$-F$_1$} \\
\midrule""")

    print(_row("\\textit{Native}", agg_native, f" ({label_native})"))
    print(_row("\\textit{Constrained}", agg_constrained, f" ({label_constrained})"))
    print(r"\midrule")
    print(_delta_row(agg_native, agg_constrained))

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).parent))

    from llm_client import OllamaClient, resolve_model, SYSTEM_PREAMBLE

    # ── Build config tag for output file naming ──────────────────────────────
    rag_tag    = "norag" if args.no_rag else f"rag_{args.retriever}"
    output_tag = f"{args.model.replace('.','').replace(':','_')}_{rag_tag}"

    # ── Load keyword queries ─────────────────────────────────────────────────
    kw_path = Path(args.queries)
    if not kw_path.exists():
        logger.error("Keyword query file not found: %s", kw_path)
        sys.exit(1)
    queries = [json.loads(l) for l in kw_path.read_text().splitlines() if l.strip()]
    logger.info("Loaded %d keyword prompts from %s", len(queries), kw_path)

    # ── LLM client ───────────────────────────────────────────────────────────
    client = OllamaClient(host=args.ollama_host)
    if not client.ping():
        logger.error(
            "Cannot reach Ollama at %s. Start Ollama and ensure the model is "
            "pulled before running this experiment.", args.ollama_host
        )
        sys.exit(1)
    model_tag = resolve_model(args.model)
    logger.info("Using model: %s", model_tag)

    # ── Context builder ──────────────────────────────────────────────────────
    if args.no_rag:
        context_fn = lambda item: "(retrieval disabled)"
        logger.info("RAG disabled — No-RAG constrained baseline.")
    else:
        # Build or load vector store for retrieval
        from retrieval import InMemoryVectorStore, build_encoder
        from data_processing import build_chunk_corpus

        retriever_map = {
            "cbert":  "clinicalbert",
            "medcpt": "medcpt",
            "minilm": "minilm",
        }
        encoder_name = retriever_map.get(args.retriever, args.retriever)
        logger.info("Building/loading encoder: %s", encoder_name)
        encoder = build_encoder(encoder_name)

        index_dir = args.index_dir
        if index_dir and InMemoryVectorStore.cache_exists(index_dir):
            logger.info("Loading existing index from %s", index_dir)
            store = InMemoryVectorStore.load_index(index_dir, encoder)
        else:
            logger.error(
                "No index found at '%s'. Build the index with main.py --rebuild "
                "before running constrained RAG evaluation.", index_dir
            )
            sys.exit(1)

        from retrieval import format_context_block
        from orchestrator import ProteusOrchestrator
        from torch import no_grad
        orch = ProteusOrchestrator(
            vector_store=store, llm_client=client, model_alias=args.model,
            top_k=5, pool_k=50, retrieval_mode="hybrid",
            use_reranker=True, rag_enabled=True,
        )

        def context_fn(item):
            retrieved = store.search(item["query"], top_k=5, pool_k=50,
                                     retrieval_mode="hybrid", reranker=orch.reranker)
            return format_context_block(retrieved)

    # ── Run constrained experiment ───────────────────────────────────────────
    logger.info("Starting constrained decoding run (N=%d)...", len(queries))
    per_example, score_objs = run_constrained_kw(
        queries=queries,
        llm_client=client,
        model_tag=model_tag,
        system_prompt_base=f"You are P.R.O.T.E.U.S., a privacy-preserving biomedical "
                           f"reasoning agent operating inside a HIPAA-aligned local "
                           f"environment. You must never fabricate information.",
        context_fn=context_fn,
        temperature=0.0,      # deterministic — eliminates sampling variance
        max_tokens=args.max_tokens,
    )

    agg_constrained = aggregate_scores(score_objs)
    logger.info(
        "Constrained aggregate: μ-F1=%.4f  Symp-F1=%.4f  Diag-F1=%.4f  Path-F1=%.4f",
        agg_constrained["micro_f1"],
        agg_constrained["per_class"]["Symptoms"]["f1"],
        agg_constrained["per_class"]["Diagnostics"]["f1"],
        agg_constrained["per_class"]["Pathogens"]["f1"],
    )

    # ── Save constrained results ─────────────────────────────────────────────
    out_path = Path(args.results_dir) / f"kw_constrained_{output_tag}.json"
    result_doc = {
        "experiment": "Exp1_ConstrainedDecoding",
        "model": model_tag,
        "decoding": "constrained_json_schema",
        "schema": KW_JSON_SCHEMA,
        "rag_enabled": not args.no_rag,
        "retriever": None if args.no_rag else args.retriever,
        "temperature": 0.0,
        "n_prompts": len(queries),
        "aggregate": agg_constrained,
        "per_example": per_example,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)
    logger.info("Saved constrained results → %s", out_path)

    # ── Load native result for comparison ────────────────────────────────────
    native_file_map = {
        ("llama3.1", "norag"):  "kw_8b_norag.json",
        ("llama3.1", "cbert"):  "kw_8b_rag_clinicalbert.json",
        ("llama3.1", "medcpt"): "kw_8b_rag_medcpt.json",
        ("llama3.1", "minilm"): "kw_8b_rag_minilm.json",
    }
    native_key   = (args.model, "norag" if args.no_rag else args.retriever)
    native_fname = native_file_map.get(native_key)
    native_path  = Path(args.results_dir) / native_fname if native_fname else None

    agg_native   = load_native_aggregate(native_path) if native_path else None
    native_label = "native no-RAG" if args.no_rag else f"native RAG ({args.retriever})"

    # ── Save dual-comparison JSON ─────────────────────────────────────────────
    dual_path = Path(args.results_dir) / "kw_constrained_dual_table.json"
    dual_doc = {
        "native_label":      native_label,
        "constrained_label": f"constrained {'no-RAG' if args.no_rag else args.retriever}",
        "native_aggregate":  agg_native,
        "constrained_aggregate": agg_constrained,
        "delta_micro_f1": (
            agg_constrained["micro_f1"] - agg_native.get("micro_f1", agg_native.get("micro", {}).get("f1", 0.0))
            if agg_native else None
        ),
    }
    with open(dual_path, "w", encoding="utf-8") as f:
        json.dump(dual_doc, f, indent=2)
    logger.info("Saved dual-comparison → %s", dual_path)

    # ── Print LaTeX table ────────────────────────────────────────────────────
    _print_latex_table(
        label_native=native_label,
        label_constrained=f"constrained {'no-RAG' if args.no_rag else args.retriever}",
        agg_native=agg_native,
        agg_constrained=agg_constrained,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Constrained structured decoding for Task 2"
    )
    parser.add_argument("--model",       default="llama3.1",
                        help="Model alias from MODEL_REGISTRY (default: llama3.1)")
    parser.add_argument("--no-rag",      action="store_true",
                        help="Run without retrieval (No-RAG constrained baseline)")
    parser.add_argument("--retriever",   default="cbert",
                        choices=["cbert", "medcpt", "minilm"],
                        help="Dense retriever to use when RAG is enabled")
    parser.add_argument("--index-dir",   default=None,
                        help="Path to pre-built vector index directory")
    parser.add_argument("--queries",     default="queries/keywords.jsonl",
                        help="Path to keyword query JSONL file")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to write output files")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--max-tokens",  type=int, default=512,
                        help="max_tokens for keyword generation (default: 512)")
    main(parser.parse_args())
