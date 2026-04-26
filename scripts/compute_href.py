"""
compute_href.py
===============
Post-hoc h_ref scoring on existing QA result JSONs.

h_ref(y) = 1 - (1/n) * sum_k [ Ent(reference, f_k) ]

where {f_k} are atomic claims decomposed from the *generated* text and
Ent is the same majority-vote three-model NLI ensemble used for h_ctx.
Unlike h_ctx, h_ref uses the gold reference answer as a single premise
(no max-pooling over chunks) and is therefore defined for No-RAG outputs.

The script re-decomposes each generated text into atomic claims via
AtomicFactExtractor (requires Ollama running), then scores every claim
against the reference with the NLI ensemble.  Pass --no-atomic-facts to
skip the LLM decomposition step and use the regex sentence-splitter
directly — this is faster and Ollama-free, at the cost of coarser claim
granularity.

Usage
-----
python compute_href.py \\
    --input  results/qa_8b_rag_cbert.json results/qa_8b_norag.json \\
    --output results/href/ \\
    [--model llama3.1] \\
    [--no-atomic-facts] \\
    [--ollama-host http://localhost:11434] \\
    [--n-resamples 10000] \\
    [--log-level INFO]

Output
------
For each input file <stem>.json the script writes
<output_dir>/<stem>_href.json containing:
  - All original top-level metadata fields (task, model, retriever, …)
  - h_ref_mean / h_ref_ci_low / h_ref_ci_high added to the "aggregate" dict
  - Per-example scores.h_ref and scores.h_ref_n_claims added to each
    entry in the "per_example" list

The per-example h_ref values from two output files can then be passed
directly to evaluation.paired_bootstrap_ci() and
evaluation.paired_permutation_test() to populate Table 2's blank row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluation import (
    AtomicFactExtractor,
    NliEnsembleHallucinationDetector,
    bootstrap_ci_single,
    split_sentences,
)
from llm_client import OllamaClient

logger = logging.getLogger("compute_href")


# ---------------------------------------------------------------------------
# Core h_ref logic
# ---------------------------------------------------------------------------

def compute_href_single(
    generated: str,
    reference: str,
    detector: NliEnsembleHallucinationDetector,
    extractor: Optional[AtomicFactExtractor],
) -> Tuple[float, int]:
    """Compute h_ref for one (generated, reference) pair.

    Parameters
    ----------
    generated:
        The model's generated answer text.
    reference:
        The gold reference answer (single premise — no max-pooling).
    detector:
        Loaded NLI ensemble.
    extractor:
        If provided, decompose *generated* into atomic claims via the LLM
        fact extractor; otherwise fall back to the regex sentence-splitter.

    Returns
    -------
    (h_ref, n_claims)
        h_ref ∈ [0, 1]; n_claims is the number of atomic claims scored.
    """
    if extractor is not None:
        claims, _ = extractor.extract(generated)
    else:
        claims = split_sentences(generated)

    if not claims:
        return 1.0, 0

    # h_ref: single reference premise per claim, no max-pooling.
    entailed: List[bool] = detector.predict_is_entailed(
        premises=[reference] * len(claims),
        hypotheses=claims,
    )
    n_supported = sum(entailed)
    h_ref = 1.0 - n_supported / len(claims)
    return h_ref, len(claims)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_result_file(
    input_path: Path,
    output_path: Path,
    detector: NliEnsembleHallucinationDetector,
    extractor: Optional[AtomicFactExtractor],
    n_resamples: int,
) -> None:
    """Enrich one QA result JSON with h_ref scores and write the output."""
    logger.info("Processing %s", input_path)
    data: Dict[str, Any] = json.loads(input_path.read_text(encoding="utf-8"))

    per_example: List[Dict[str, Any]] = data.get("per_example", [])
    if not per_example:
        logger.warning("%s contains no per_example entries — skipping", input_path.name)
        return

    h_ref_values: List[float] = []

    for i, ex in enumerate(per_example, start=1):
        generated: str = ex.get("generated", "")
        reference: str = ex.get("reference", "")

        if not reference:
            logger.warning(
                "[%d/%d] No reference field — h_ref set to NaN", i, len(per_example)
            )
            ex.setdefault("scores", {}).update({"h_ref": float("nan"), "h_ref_n_claims": 0})
            continue

        h_ref, n_claims = compute_href_single(generated, reference, detector, extractor)
        logger.info("[%d/%d] h_ref=%.4f  n_claims=%d", i, len(per_example), h_ref, n_claims)

        ex.setdefault("scores", {}).update({
            "h_ref": round(h_ref, 6),
            "h_ref_n_claims": n_claims,
        })
        h_ref_values.append(h_ref)

    # Aggregate: mean + 95 % bootstrap CI (seed=42, configurable n_resamples).
    agg: Dict[str, Any] = data.setdefault("aggregate", {})
    if h_ref_values:
        arr = np.array(h_ref_values, dtype=float)
        ci_lo, ci_hi = bootstrap_ci_single(arr, n_resamples=n_resamples)
        agg.update({
            "h_ref_mean":     round(float(arr.mean()), 6),
            "h_ref_ci_low":   round(float(ci_lo), 6),
            "h_ref_ci_high":  round(float(ci_hi), 6),
        })
        logger.info(
            "  %s  h_ref mean=%.4f  95%%CI=[%.4f, %.4f]  (n=%d, %d resamples)",
            input_path.stem, agg["h_ref_mean"], ci_lo, ci_hi,
            len(h_ref_values), n_resamples,
        )
    else:
        agg.update({
            "h_ref_mean": float("nan"),
            "h_ref_ci_low": float("nan"),
            "h_ref_ci_high": float("nan"),
        })
        logger.warning("No valid h_ref values computed for %s", input_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Wrote enriched results → %s", output_path)


# ---------------------------------------------------------------------------
# Table 2 helper: print paired stats for the h_ref contrast
# ---------------------------------------------------------------------------

def print_paired_stats(
    path_rag: Path,
    path_norag: Path,
    n_resamples: int,
) -> None:
    """Print the paired bootstrap CI and permutation p-value for the
    8B-RAG-vs-8B-NoRAG h_ref contrast (Table 2, currently-blank row).

    Imported lazily so the script works even if this helper is not called.
    """
    from evaluation import paired_bootstrap_ci, paired_permutation_test  # noqa: PLC0415

    def _load_href(path: Path) -> List[float]:
        data = json.loads(path.read_text(encoding="utf-8"))
        vals = []
        for ex in data.get("per_example", []):
            v = ex.get("scores", {}).get("h_ref")
            if v is not None and not (isinstance(v, float) and v != v):  # exclude NaN
                vals.append(float(v))
        return vals

    a = _load_href(path_rag)
    b = _load_href(path_norag)
    if len(a) != len(b) or not a:
        logger.error(
            "Cannot compute paired stats: n_rag=%d n_norag=%d — lengths must match",
            len(a), len(b),
        )
        return

    boot = paired_bootstrap_ci(a, b, n_resamples=n_resamples)
    perm = paired_permutation_test(a, b, n_permutations=n_resamples)

    print("\n=== Table 2 · 8B-RAG vs 8B-NoRAG · h_ref ===")
    print(f"  anchor  mean h_ref : {sum(a)/len(a):.4f}  ({path_rag.stem})")
    print(f"  compare mean h_ref : {sum(b)/len(b):.4f}  ({path_norag.stem})")
    print(f"  Δ (anchor − compare): {boot['mean_diff']:+.4f}")
    print(f"  95% bootstrap CI    : [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
    print(f"  permutation p-value : {perm['p_value']:.4f}")
    print(f"  significant (CI∩p)  : {boot['significant'] and perm['p_value'] < 0.05}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute_href",
        description="Post-hoc h_ref NLI scoring on existing QA result JSONs.",
    )
    p.add_argument(
        "--input", nargs="+", required=True, metavar="RESULT_JSON",
        help="One or more qa_*.json result files to enrich.",
    )
    p.add_argument(
        "--output", required=True, metavar="OUTPUT_DIR",
        help="Directory to write <stem>_href.json output files.",
    )
    p.add_argument(
        "--model", default="llama3.1",
        help="Model alias for AtomicFactExtractor (default: llama3.1).",
    )
    p.add_argument(
        "--no-atomic-facts", action="store_true",
        help="Skip LLM decomposition; use the regex sentence-splitter instead. "
             "Faster and Ollama-free, but produces sentence-level rather than "
             "atomic-claim-level granularity.",
    )
    p.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Ollama REST endpoint (ignored when --no-atomic-facts is set).",
    )
    p.add_argument(
        "--n-resamples", type=int, default=10_000,
        help="Bootstrap/permutation resamples for CIs (default: 10 000).",
    )
    p.add_argument(
        "--paired-stats", nargs=2, metavar=("RAG_HREF_JSON", "NORAG_HREF_JSON"),
        help="After scoring, print paired bootstrap CI and permutation p-value "
             "for the Table 2 h_ref contrast.  Pass the two *_href.json output "
             "files produced by this script.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading NLI ensemble on %s …", device)
    detector = NliEnsembleHallucinationDetector(device=device)

    extractor: Optional[AtomicFactExtractor] = None
    if not args.no_atomic_facts:
        llm = OllamaClient(host=args.ollama_host)
        if not llm.ping():
            logger.error(
                "Ollama not reachable at %s — use --no-atomic-facts to fall back "
                "to sentence splitting without Ollama.",
                args.ollama_host,
            )
            return 1
        extractor = AtomicFactExtractor(llm_client=llm, model_alias=args.model)
        logger.info("AtomicFactExtractor ready (model=%s)", args.model)
    else:
        logger.info(
            "--no-atomic-facts: using regex sentence-splitter for claim decomposition"
        )

    output_dir = Path(args.output)
    for input_str in args.input:
        input_path = Path(input_str)
        if not input_path.exists():
            logger.error("Input file not found: %s — skipping", input_path)
            continue
        output_path = output_dir / f"{input_path.stem}_href.json"
        process_result_file(
            input_path, output_path, detector, extractor, args.n_resamples
        )

    if args.paired_stats:
        rag_path, norag_path = Path(args.paired_stats[0]), Path(args.paired_stats[1])
        if rag_path.exists() and norag_path.exists():
            print_paired_stats(rag_path, norag_path, args.n_resamples)
        else:
            logger.error(
                "--paired-stats files not found: %s, %s", rag_path, norag_path
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
